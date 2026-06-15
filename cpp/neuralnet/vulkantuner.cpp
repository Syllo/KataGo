#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkankernels.h"
#include "../neuralnet/vulkanlayers.h"
#include "../neuralnet/vulkantuner_shared.h"

using namespace std;
using namespace VulkanHelpers;

// ============================================================================
// Shared bench-support helpers (also used by the VulkanKernels::*::bench
// implementations in vulkantunebench.cpp).
// ============================================================================

namespace VulkanTuner {

  // True when (tm,tn,tk) is a fragment shape the device actually reports.
  bool coopmatShapeIsSupported(const std::vector<CoopmatShape>& shapes, int32_t tm, int32_t tn, int32_t tk) {
    for(const auto& shape: shapes)
      if(shape.m == (uint32_t)tm && shape.n == (uint32_t)tn && shape.k == (uint32_t)tk)
        return true;
    return false;
  }

  // Derive the NHWC Winograd transform/untransform data layout implied by the
  // currently-selected Winograd GEMM variant (tile sizes, A packing) plus
  // fp16Storage. Resolves the variant exactly as the runtime will (the
  // device-info overload masks filter-disabled variants out of a scratch copy).
  WinogradTransformBenchLayout makeWinogradTransformBenchLayout(
    const VulkanTuneParams& cfg,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute) {
    const bool fp16 = fp16Storage && fp16Compute;
    const int64_t variant = resolveWinogradGemmVariant(deviceInfo, cfg, fp16);

    WinogradTransformBenchLayout layout;
    const LayerPaddingContract contract = selectedWinogradPaddingContract(cfg, variant);
    layout.mAlignment = std::max(1, contract.m);
    layout.nAlignment = std::max(1, contract.n);
    layout.kAlignment = std::max(1, contract.kPaddable ? contract.k : 1);
    layout.packedBM = std::max(1, (int)cfg.winogradGemmM);
    layout.packedBK = std::max(1, (int)cfg.winogradGemmK);
    layout.packedAPadWords = WINOGRAD_ROW_MAJOR_A_PAD_WORDS;
    switch(variant) {
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16:
        layout.packedBM = std::max(1, (int)cfg.coopmat2AccF16BM);
        layout.packedBK = std::max(1, (int)cfg.coopmat2AccF16BK);
        layout.packedAPadWords = WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
        break;
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT2:
        layout.packedBM = std::max(1, (int)cfg.coopmat2BM);
        layout.packedBK = std::max(1, (int)cfg.coopmat2BK);
        layout.packedAPadWords = WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
        break;
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16:
        layout.packedBM = std::max(1, (int)cfg.coopmat1AccF16BM);
        layout.packedBK = std::max(1, (int)cfg.coopmat1AccF16BK);
        layout.packedAPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
        break;
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT1:
        layout.packedBM = std::max(1, (int)cfg.coopmat1BM);
        layout.packedBK = std::max(1, (int)cfg.coopmat1BK);
        layout.packedAPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
        break;
      case VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16:
        layout.packedBM = std::max(1, (int)cfg.dot2AccF16BM);
        layout.packedBK = VulkanKernels::DOT2_BK;
        break;
      case VulkanTuner::TUNED_WINOGRAD_DOT2:
        layout.packedBM = std::max(1, (int)cfg.dot2BM);
        layout.packedBK = VulkanKernels::DOT2_BK;
        break;
      case VulkanTuner::TUNED_WINOGRAD_TILED:
        break;
      default:
        ASSERT_UNREACHABLE;
    }
    return layout;
  }

  // Round a wall-clock seconds measurement to a stored microsecond field,
  // clamped to the int32 range; 0 when the measurement is unusable.
  int32_t secondsToMicros(double seconds) {
    if(!(seconds > 0.0) || !std::isfinite(seconds))
      return int32_t(0);
    const double micros = std::round(seconds * 1.0e6);
    return (int32_t)std::max(1.0, std::min(micros, (double)std::numeric_limits<int32_t>::max()));
  }

  // Self-apply one candidate's chosen (field, value) pairs to a config.
  void applyCandidate(const TunedCandidate& cand, VulkanTuneParams& cfg) {
    for(const auto& e: cand.values)
      cfg.*(e.field) = e.value;
  }

}  // namespace VulkanTuner

// ============================================================================
// TU-local helpers
// ============================================================================

namespace {

  // Maximum normalized RMSE a candidate output may show against the reference
  // before the tuning result is rejected (2% of the reference magnitude).
  constexpr double MAX_RMSE_TOLERANCE = 0.02;

}  // namespace

// Tuning machinery (Vulkan-API-dependent): TuningContext lifecycle, the search
// loop, the kernel factory, and the VulkanKernels::*::bench definitions.
// Folded in from the former vulkantunable.cpp; TuningContext::makeInputBufFP/
// downloadFloatsFP live in vulkanbackend.cpp where its base class does.
// ============================================================================

namespace VulkanTuner {

  // ---- Shared helpers ----

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

    struct GemmStridedModeCase {
      int M;
      int N;
      int K;
      int occurrences;
      int overwriteOccurrences;
      int addToOutputOccurrences;
      double weightedWork;
    };

    static void
    addGemmStridedModeCase(std::vector<GemmStridedModeCase>& cases, int M, int rawN, int rawK, bool addToOutput) {
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
          addGemmStridedModeCase(
            outCases, paddedSpatialSize, attn->outProj.outChannels, attn->outProj.inChannels, true);
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

    // Advance a mixed-radix counter over the Cartesian product of
    // params[i].candidates. Returns false when the counter wraps (all
    // combinations have been visited).
    static bool advanceParamOdometer(std::vector<size_t>& idx, ArrayView<TunableParam> params) {
      size_t pos = 0;
      while(pos < params.size()) {
        idx[pos]++;
        if(idx[pos] < params[pos].candidates.size())
          return true;
        idx[pos] = 0;
        pos++;
      }
      return false;
    }

    // A workgroup narrower than one subgroup leaves lanes idle every dispatch and
    // is not expected to win the lowest-time selection, so benching it is usually
    // pure waste. We prune those candidates from a sweep, but only when the sweep
    // can otherwise reach a full subgroup, so a kernel whose entire candidate set
    // is sub-subgroup by design (e.g. a reduction pinned below the subgroup width)
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
        done = !advanceParamOdometer(idx, params);
      }
      return anyReachesSubgroup ? subgroupSize : 0;
    }

    static bool workgroupBelowFloor(const TunableKernel& kernel, const VulkanTuneParams& cfg, uint32_t subgroupFloor) {
      if(subgroupFloor == 0)
        return false;
      uint32_t threads = kernel.workgroupThreads(cfg);
      return threads > 0 && threads < subgroupFloor;
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
      return secondsToMicros(stats.weightedTime);
    }

    constexpr size_t TUNER_CONFIRM_K = 6;
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
      selected.reserve(std::min(candidateCount, TUNER_CONFIRM_K + prefixCount));
      auto addIfNew = [&](size_t idx) {
        if(!contains(selected, idx))
          selected.push_back(idx);
      };
      for(size_t i = 0; i < ranked.size() && i < TUNER_CONFIRM_K; i++)
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

  static std::vector<std::unique_ptr<TunableKernel>>
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
    kernels.push_back(
      std::make_unique<VulkanKernels::NchwToNhwcTuner>(std::max(1, batchSize), paddedSpatialSize, layoutChannels));

    int gpoolC3 = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels * 3 : 192;
    kernels.push_back(std::make_unique<VulkanKernels::GemmDirectFP32>(batchSize > 0 ? batchSize : 1, gpoolC3, gpoolC3));
    int gpoolC = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels : 32;
    kernels.push_back(std::make_unique<VulkanKernels::GPoolReductionNhwc>(batchSize, gpoolC, nnXLen * nnYLen));
    kernels.push_back(std::make_unique<VulkanKernels::ValueHeadPoolNhwc>(batchSize, gpoolC, nnXLen * nnYLen));
    // NHWC Winograd transform/untransform (non-coopmat fallback conv path).
    // Only actually benched when their own tuned bits are in the missing mask
    // (requiredKernelMask omits them on coopmat-capable devices, where this
    // path is never selected), so it's safe to always add these here.
    int wgChannelsNhwc = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
    int wgInChannelsNhwc = std::max({modelDesc->numInputChannels, trunk.trunkNumChannels, trunk.midNumChannels, 64});
    kernels.push_back(
      std::make_unique<VulkanKernels::WinogradTransformNhwc>(batchSize, wgInChannelsNhwc, nnXLen, nnYLen));
    kernels.push_back(
      std::make_unique<VulkanKernels::WinogradUntransformNhwc>(batchSize, wgChannelsNhwc, nnXLen, nnYLen));
    if(!modelHasTransformer(modelDesc))
      return kernels;

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
    int fcGpoolC3 = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels * 3 : 192;
    kernels.push_back(
      std::make_unique<VulkanKernels::GemmDirectFP32>(batchSize > 0 ? batchSize : 1, fcGpoolC3, fcGpoolC3));

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
          batchSize,
          seqLen,
          benchHeadDim,
          benchVHeadDim,
          benchNumHeads,
          benchNumKVHeads,
          useRope,
          learnableRope,
          nnXLen * nnYLen));
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
      out << endl;
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
          applyCandidate(cand, cfg);
          KernelBench b = kernel.bench(ctx, cfg, iters);
          if(!(b.ok && b.kernelsPerSecond > 0.0))
            return ConfirmationSample{false, 0.0, 0.0, false, false};
          double rmse = normalizedRmse(ref.output, b.output);
          bool rmseOk = rmse <= MAX_RMSE_TOLERANCE;
          return ConfirmationSample{rmseOk, b.kernelsPerSecond, 0.0, true, rmseOk};
        });

      std::vector<TunedCandidate> confirmed;
      confirmed.reserve(confirmIndexes.size());
      double bestConfirmedKps = -1.0;
      for(size_t pos = 0; pos < confirmResults.size(); pos++) {
        const ConfirmationResult& result = confirmResults[pos];
        TunedCandidate cand = candidates[result.candidateIndex];
        VulkanTuneParams cfg = seed;
        applyCandidate(cand, cfg);
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

  static std::vector<TunedCandidate> tuneOne(
    TunableKernel& kernel,
    const VulkanTuneParams& seed,
    TuningContext& ctx,
    int iters,
    std::ostream& out,
    bool verboseTuner) {
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
            bool within = rmse <= MAX_RMSE_TOLERANCE;
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
      done = !advanceParamOdometer(idx, params);
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
      applyCandidate(scored[0], selectedCfg);
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
          if(rmse > MAX_RMSE_TOLERANCE)
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

        done = !advanceParamOdometer(idx, params);
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
      bool tuned = tuneTiledMultiShape<KernelT>(
        kernelName, modeCases, seed, ctx, iters, out, verboseTuner, tunedOut, makeProblem, makeKernel, printCase);
      if(tuned)
        tunedOut.markKernelTuned({TUNED_GEMM_STRIDED_TILED});
      return tuned;
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
      bool tuned = tuneTiledMultiShape<KernelT>(
        kernelName, modeCases, seed, ctx, iters, out, verboseTuner, tunedOut, makeProblem, makeKernel, printCase);
      if(tuned)
        tunedOut.markKernelTuned({TUNED_WINOGRAD_TILED});
      return tuned;
    }

    struct Dot2Tile {
      int32_t workgroupSize, bm, bn, sgm, sgn, sgmIter, tm, tn, subgroupSize;
    };

    // ---- Tile -> VulkanTuneParams field mapping ----
    //
    // Every tile-shaped .inc field group lists its fields in the same order as
    // the corresponding tile struct members (verified against
    // vulkantuneparams_fields.inc). applyTileFields zips a group's member-pointer
    // list with a tile's scalars, so the applyWinograd*Tile / applyConv*Tile
    // functions and the Ops traits' applyGemm/applyConv all collapse to one line.
    // Each list must stay in tile-member order; the count static_asserts keep
    // group size and tile member count from drifting.

    template<size_t N>
    void applyTileFields(
      VulkanTuneParams& cfg,
      int32_t VulkanTuneParams::* const (&fields)[N],
      const std::array<int32_t, N>& scalars) {
      for(size_t i = 0; i < N; i++)
        cfg.*fields[i] = scalars[i];
    }

    // Tile scalars, in tile-member order (must match the k*TileFields arrays).
    // coopmatTileScalars/coopmat2TileScalars are defined after their structs below.
    std::array<int32_t, 9> dot2TileScalars(const Dot2Tile& t) {
      return {t.workgroupSize, t.bm, t.bn, t.sgm, t.sgn, t.sgmIter, t.tm, t.tn, t.subgroupSize};
    }

    // Hand-tuned dot2 tile geometries. The subgroup size and workgroup size are
    // NOT stored here — they are derived from the device by makeDot2Candidates:
    // on a device that can pin a subgroup size (VK_EXT_subgroup_size_control),
    // the tiling uses the device's reported physical subgroup size so the
    // logical register blocks align to real hardware subgroups; otherwise it
    // falls back to the classic 32-lane blocks. Each geometry must tile the
    // BMxBN block with an integer number of subgroups (bm%sgm==0, bn%sgn==0).
    struct Dot2Geometry {
      int32_t bm, bn, sgm, sgn, sgmIter, tm, tn;
    };

    static const Dot2Geometry kDot2Geometries[] = {
      {64, 64, 32, 32, 2, 4, 2},    // ggml F16 default
      {32, 32, 32, 32, 2, 2, 2},    // small tile (small matmuls)
      {128, 128, 64, 64, 2, 4, 4},  // large tile (big trunk GEMMs)
      {128, 64, 64, 32, 2, 4, 2},   // tall tile
      {64, 128, 32, 64, 2, 4, 4},   // wide tile
      // workgroupSize=64 (2-subgroup) thin tiles. Halving one tile dimension roughly doubles
      // the workgroup count, which helps two device classes: the strided path (gz =
      // batchSize, often 1, so a 64x64 tile underfills a wide GPU) and weak iGPUs/APUs
      // (few CUs, register/LDS-starved — more co-resident 2-subgroup workgroups hide
      // latency better than one 4-subgroup tile). Both bm/bn divide the sweep LCM (128),
      // so adding them does not change the LCM-padded bench dims of the other tiles.
      {32, 64, 32, 32, 2, 4, 2},  // thin-M tile
      {64, 32, 32, 32, 2, 4, 2},  // thin-N tile
    };

    // Build dot2 tile candidates for this device: keep the hand-tuned geometries
    // above but derive the subgroup size (and the workgroup size it implies) from
    // the device, so logical register blocks line up with the physical subgroup
    // when the device can pin one. Mirrors makeCoopmatCandidates.
    std::vector<Dot2Tile> makeDot2Candidates(const TuningContext& ctx) {
      std::vector<Dot2Tile> out;
      if(ctx.dev == nullptr)
        return out;
      const uint32_t subgroupSize =
        ctx.dev->info.canRequireComputeSubgroupSize(ctx.dev->info.subgroupSize) ? ctx.dev->info.subgroupSize : 32u;
      for(const Dot2Geometry& g: kDot2Geometries) {
        const int32_t workgroupSize = (g.bm / g.sgm) * (g.bn / g.sgn) * (int32_t)subgroupSize;
        Dot2Tile t{workgroupSize, g.bm, g.bn, g.sgm, g.sgn, g.sgmIter, g.tm, g.tn, (int32_t)subgroupSize};
        if(
          VulkanKernels::WinogradGemmDot2::isConfigSupported(
            t.workgroupSize, t.bm, t.bn, t.sgm, t.sgn, t.sgmIter, t.tm, t.tn, t.subgroupSize))
          out.push_back(t);
      }
      return out;
    }

    std::string dot2TileToString(const Dot2Tile& t) {
      return Global::strprintf(
        "[%d,%d,%d,%d,%d,%d,%d,%d,%d]",
        t.workgroupSize,
        t.bm,
        t.bn,
        t.sgm,
        t.sgn,
        t.sgmIter,
        t.tm,
        t.tn,
        t.subgroupSize);
    }

    bool dot2TilesEqual(const Dot2Tile& a, const Dot2Tile& b) {
      return a.workgroupSize == b.workgroupSize && a.bm == b.bm && a.bn == b.bn && a.sgm == b.sgm && a.sgn == b.sgn &&
             a.sgmIter == b.sgmIter && a.tm == b.tm && a.tn == b.tn && a.subgroupSize == b.subgroupSize;
    }

    void applyWinogradDot2Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      applyTileFields(cfg, kWinogradDot2TileFields, dot2TileScalars(t));
    }

    void applyWinogradDot2AccF16Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      applyTileFields(cfg, kWinogradDot2AccF16TileFields, dot2TileScalars(t));
    }

    // Measure one Winograd GEMM variant's aggregate (occurrences-weighted) time across the
    // model's winograd mode cases and store it in the variant's TimeUs field. Each case is
    // RMSE-gated against a tiled reference, so a variant whose output diverges from the
    // reference never gets a nonzero score. This fills in a missing TimeUs after a tile
    // sweep; the family/accumulator selection itself happens at runtime via
    // resolveWinogradGemmVariant, so there is no incumbent-vs-challenger decision here.
    template<typename ChallengeKernel, int32_t VulkanTuneParams::* TimeUsField>
    void tuneWinogradGemmVariantModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      string_view challengerName,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> wCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(wCases.empty())
        return;

      ModeSelectStats stats;
      double maxRmse = 0.0;

      out << endl
          << "VulkanTuner: winogradGemm " << challengerName << " mode-select cases (" << wCases.size() << ") ..."
          << endl;
      for(size_t i = 0; i < wCases.size(); i++) {
        const WinogradModeCase& cs = wCases[i];
        const auto pad = padDims(cs.M, cs.N, cs.K, ChallengeKernel::layerPaddingContract(tunedConfig));
        ChallengeKernel probe(pad.m, pad.n, pad.k, cs.numBatches);
        KernelBench bench = probe.bench(ctx, tunedConfig, iters);
        double tflops = probe.estimatedTflops(tunedConfig, bench.kernelsPerSecond);
        bool ok = bench.ok && bench.kernelsPerSecond > 0.0;
        double caseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(ok) {
          VulkanKernels::WinogradGemm refProbe(pad.m, pad.n, pad.k, cs.numBatches);
          KernelBench refBench = refProbe.bench(ctx, tunedConfig, iters);
          ok = refBench.ok;
          if(ok) {
            rmse = normalizedRmse(refBench.output, bench.output);
            maxRmse = std::max(maxRmse, rmse);
            ok = rmse <= MAX_RMSE_TOLERANCE;
          }
        }
        if(ok)
          caseWeightedTime =
            stats.recordCase(true, bench, probe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);
        else
          stats.allOk = false;

        if(shouldPrintTunerResult(verboseTuner, false, ok)) {
          out << "VulkanTuner:   winogradGemm " << challengerName << " case#" << i;
          if(bench.ok && bench.kernelsPerSecond > 0.0) {
            out << " " << challengerName << "_kps=" << bench.kernelsPerSecond;
            if(tflops > 0.0)
              out << " " << challengerName << "_tflops=" << tflops;
            out << " " << challengerName << "_weighted_time=" << caseWeightedTime << " dims=" << pad.m << "x" << pad.n
                << "x" << pad.k;
          } else
            out << " " << challengerName << "=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " result=" << (ok ? "accepted" : "rejected") << endl;
        }
      }

      tunedConfig.*TimeUsField = weightedTimeToMicros(stats);
      out << "VulkanTuner: winogradGemm " << challengerName << " mode-select"
          << " weighted_time=" << stats.weightedTime << " tflops=" << stats.tflops() << " max_rmse=" << maxRmse << endl;
    }

    // ---- Coopmat (VK_KHR_cooperative_matrix) mode-select ----

    struct CoopmatTile {
      int32_t workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize;
    };

    std::array<int32_t, 10> coopmatTileScalars(const CoopmatTile& t) {
      return {t.workgroupSize, t.bm, t.bn, t.bk, t.sgm, t.sgn, t.tm, t.tn, t.tk, t.subgroupSize};
    }

    std::string coopmatTileToString(const CoopmatTile& t) {
      return Global::strprintf(
        "[bs=%d bm=%d bn=%d bk=%d sgm=%d sgn=%d tm=%d tn=%d tk=%d subgroup=%d]",
        t.workgroupSize,
        t.bm,
        t.bn,
        t.bk,
        t.sgm,
        t.sgn,
        t.tm,
        t.tn,
        t.tk,
        t.subgroupSize);
    }

    // Build coopmat tile candidates from the device's reported fragment shapes.
    // TM/TN/TK come verbatim from each shape; only the block/subgroup tiling is varied.
    // Every candidate satisfies coopmatConfigSupported by construction.
    std::vector<CoopmatTile> makeCoopmatCandidates(const TuningContext& ctx, const std::vector<CoopmatShape>& shapes) {
      std::vector<CoopmatTile> out;
      if(ctx.dev == nullptr)
        return out;
      const uint32_t subgroupSize = ctx.dev->info.subgroupSize > 0 ? ctx.dev->info.subgroupSize : 32u;
      for(const auto& shape: shapes) {
        const int32_t tm = (int32_t)shape.m;
        const int32_t tn = (int32_t)shape.n;
        const int32_t tk = (int32_t)shape.k;
        if(tm <= 0 || tn <= 0 || tk <= 0)
          continue;
        // A deliberately small portfolio of complete geometries, rather than a
        // Cartesian product of independently plausible dimensions. The first
        // three are llama.cpp's coopmat1 small/medium/large geometries. The
        // next two are the 64x128 and 128x64 CTA shapes common in CUTLASS tensor
        // core kernels. The remaining profiles retain a small set of KataGo
        // winners and structural fallbacks from tuning runs.
        //
        // Values are physical dimensions, not multipliers: that makes the
        // llama.cpp 64x64 subgroup tile work for both 16x16 and 16x8 device
        // fragments. A profile is used only when every dimension is an exact
        // multiple of the reported fragment shape. Existing device and kernel
        // validation below remains authoritative.
        struct CoopmatProfile {
          int32_t sgm, sgn, bm, bn, bk;
        };
        static const CoopmatProfile profiles[] = {
          // llama.cpp coopmat1: small, medium, large.
          {32, 32,  32,  32, 16},
          {32, 32,  64,  64, 16},
          {64, 64, 128, 128, 16},
          // CUTLASS-style asymmetric CTA tiles.
          {32, 64,  64, 128, 32},
          {64, 32, 128,  64, 32},
          // KataGo small-tile winners and structural fallbacks. These retain
          // a short-K, long-K, and two-subgroup variant without carrying every
          // close result from one T4 run into every other device's search.
          {16, 16,  16,  16, 16},
          {16, 16,  16,  16, 96},
          {16, 16,  16,  32, 16},
          {16, 16,  16,  32, 32},
          // F16 Winograd winner and its essentially tied BK=32 alternative:
          // the same 32x32 output as llama's small profile, but distributed
          // over four 16x16 subgroup tiles.
          {16, 16,  32,  32, 16},
          // The third K-depth is a distinct fast path for 16x8x8 Winograd
          // fragments on larger models (BK=3*TK there).
          {16, 16,  32,  32, 24},
          {16, 16,  32,  32, 32},
          // A fast 16x8-fragment fallback from the strided sweeps.
          {32,  8,  64,  16, 32},
        };
        constexpr int maxBm = 384;
        constexpr int maxBn = 512;
        constexpr int maxSubgroupsPerWorkgroup = 8;
        for(const CoopmatProfile& p: profiles) {
          if(
            p.sgm % tm != 0 || p.sgn % tn != 0 || p.bm % p.sgm != 0 || p.bn % p.sgn != 0 ||
            p.bk % tk != 0
          )
            continue;
          if(p.sgm > 64 || p.sgn > 64 || p.bm > maxBm || p.bn > maxBn)
            continue;
          const int32_t numSubgroups = (p.bm / p.sgm) * (p.bn / p.sgn);
          if(numSubgroups > maxSubgroupsPerWorkgroup)
            continue;
          const int32_t workgroupSize = numSubgroups * (int32_t)subgroupSize;
          CoopmatTile t{workgroupSize, p.bm, p.bn, p.bk, p.sgm, p.sgn, tm, tn, tk, (int32_t)subgroupSize};
          if(
            VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
              t.workgroupSize, t.bm, t.bn, t.bk, t.sgm, t.sgn, t.tm, t.tn, t.tk, t.subgroupSize))
            out.push_back(t);
        }
      }
      return out;
    }

    int roundUpCoopmatDim(int v, int tile) {
      return ((v + tile - 1) / tile) * tile;
    }

    void applyWinogradCoopmatTile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      applyTileFields(cfg, kWinogradCoopmat1TileFields, coopmatTileScalars(t));
    }

    void applyWinogradCoopmatAccF16Tile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      applyTileFields(cfg, kWinogradCoopmat1AccF16TileFields, coopmatTileScalars(t));
    }

    bool coopmatTilesEqual(const CoopmatTile& a, const CoopmatTile& b) {
      return a.workgroupSize == b.workgroupSize && a.bm == b.bm && a.bn == b.bn && a.bk == b.bk && a.sgm == b.sgm &&
             a.sgn == b.sgn && a.tm == b.tm && a.tn == b.tn && a.tk == b.tk && a.subgroupSize == b.subgroupSize;
    }

    template<typename Ops>
    // Returns true when an aligned tile was selected for this conv size, i.e.
    // the coopmat implicit-GEMM kernel is actually usable for this model's
    // channels/spatial dims. This is distinct from markConvValid (which records
    // "attempted" and always sets the flag): the caller uses the return value to
    // decide whether the Winograd fallback needs tuning.
    bool tuneNhwcConvSweep(
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
        return false;
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
            << convSize << "x" << convSize
            << " layers; keeping "
               "defaults."
            << endl;
        Ops::markConvValid(tunedConfig, convSize);
        return false;
      }

      const Conv3x3NhwcModeCase& trimCase = modeCases[0];
      std::vector<Tile> sweepCandidates =
        Ops::trimConv(eligible, paddedSpatialSize, trimCase.outChannels, convSize * convSize * trimCase.inChannels);
      if(sweepCandidates.empty()) {
        Ops::markConvValid(tunedConfig, convSize);
        return false;
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
            if(rmse > MAX_RMSE_TOLERANCE)
              return rejectAggregateCandidate(metrics, "rmse_exceeded");

            const double latencyWeight = batchSize == 1 && problemBatchSize > 1 ? 4.0 : 1.0;
            const double caseSeconds = latencyWeight * (double)cs.occurrences / b.kernelsPerSecond;
            metrics.weightedTime += caseSeconds;
            const double flops = 2.0 * batchSize * nnXLen * nnYLen * cs.outChannels * (double)(convSize * convSize) *
                                 cs.inChannels * cs.occurrences;
            aggregateFlops += latencyWeight * flops;
            aggregateSeconds += caseSeconds;
          }
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      out << "VulkanTuner: " << Ops::convLabel(convSize) << " tile sweep candidates=" << sweepCandidates.size()
          << " raw_candidates=" << candidates.size() << " cases=" << modeCases.size() << " batches=[1,"
          << std::max(1, problemBatchSize) << "] batch1_weight=4"
          << " trim_case M=" << paddedSpatialSize << " N=" << trimCase.outChannels
          << " K=" << convSize * convSize * trimCase.inChannels << endl;
      AggregateCandidateSweepResult<Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{Ops::convLabel(convSize)},
        TunerStreamText{Ops::convLabel(convSize)},
        TunerStreamText{Ops::convLabel(convSize), " tile sweep"},
        "tile",
        TunerStreamText{
          "VulkanTuner: ", Ops::convLabel(convSize), " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_K,
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
      return result.found;
    }

    // ---- Coopmat2 (VK_NV_cooperative_matrix2) tuning ----
    // Coopmat2Tile, coopmat2TileMatchesShape and coopmat2TileSupported are
    // shared via vulkantuner_shared.h.

    std::array<int32_t, 4> coopmat2TileScalars(const Coopmat2Tile& t) {
      return {t.workgroupSize, t.bm, t.bn, t.bk};
    }

    std::string coopmat2TileToString(const Coopmat2Tile& t) {
      return Global::strprintf("bs=%d BM=%d BN=%d BK=%d", t.workgroupSize, t.bm, t.bn, t.bk);
    }

    bool coopmat2TilesEqual(const Coopmat2Tile& a, const Coopmat2Tile& b) {
      return a.workgroupSize == b.workgroupSize && a.bm == b.bm && a.bn == b.bn && a.bk == b.bk;
    }

    enum class Coopmat2CandidateFamily {
      StridedGemm,
      ImplicitConv,
      Winograd,
    };

    // Build coopmat2 tile candidates from the device's reported flexible-dimension
    // granularities. The compact profile catalog intentionally uses shape-relative
    // multipliers, not absolute T4 tile sizes, so every candidate remains appropriate
    // to the device. Each kernel family receives only the measured geometry classes
    // relevant to its workload, rather than benchmarking one union catalog for every
    // strided-GEMM, convolution, and Winograd sweep.
    // BK is capped at 64: with this small catalog, the deeper-K profiles are cheap
    // to test for all families. isConfigSupported / coopmat2TileSupported / maxDim
    // prune device-inappropriate profiles.
    std::vector<Coopmat2Tile>
    makeCoopmat2Candidates(
      const TuningContext& ctx,
      const std::vector<Coopmat2FlexShape>& shapes,
      Coopmat2CandidateFamily family,
      int maxBk = 64) {
      std::vector<Coopmat2Tile> out;
      if(ctx.dev == nullptr)
        return out;
      const VulkanDeviceInfo& info = ctx.dev->info;
      auto add = [&](Coopmat2Tile t) {
        if(!VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(t.workgroupSize, t.bm, t.bn, t.bk))
          return;
        if(!coopmat2TileSupported(info, shapes, t))
          return;
        for(const Coopmat2Tile& x: out)
          if(x.workgroupSize == t.workgroupSize && x.bm == t.bm && x.bn == t.bn && x.bk == t.bk)
            return;
        out.push_back(t);
      };
      struct Profile {
        int bmMul;
        int bnMul;
        int bkMul;
      };
      // Exact b7/b11 raw winners, expressed in the device's advertised
      // M/N/K granularities. Each list also spans both accumulator precisions.
      static const Profile stridedProfiles[] = {
        // CUTLASS balanced/asymmetric classes and llama.cpp's deep-K small
        // GEMM class keep the shortlist useful beyond the measured T4 runs.
        {2, 2, 1}, {4, 4, 2}, {4, 8, 2}, {8, 4, 2}, {4, 4, 4},
        // KataGo b7/b11 winners.
        {3, 1, 2}, {3, 4, 2}, {3, 8, 1}, {4, 2, 2},
      };
      static const Profile convProfiles[] = {
        {2, 2, 4}, {4, 3, 2},
      };
      static const Profile winogradProfiles[] = {
        {1, 4, 1}, {1, 6, 1}, {6, 3, 1},
      };
      ArrayView<Profile> profiles;
      switch(family) {
        case Coopmat2CandidateFamily::StridedGemm: profiles = stridedProfiles; break;
        case Coopmat2CandidateFamily::ImplicitConv: profiles = convProfiles; break;
        case Coopmat2CandidateFamily::Winograd: profiles = winogradProfiles; break;
        default: ASSERT_UNREACHABLE;
      }
      for(const Coopmat2FlexShape& s: shapes) {
        int maxDim =
          info.coopmat2MaxFlexDimension > 0 ? (int)std::min<uint32_t>(info.coopmat2MaxFlexDimension, 256) : 256;
        for(const Profile& p: profiles) {
          int bm = (int)s.mGranularity * p.bmMul;
          int bn = (int)s.nGranularity * p.bnMul;
          int bk = (int)s.kGranularity * p.bkMul;
          if(bm > maxDim || bn > maxDim || bk > maxDim || bk > maxBk)
            continue;
          add(Coopmat2Tile{(int32_t)s.workgroupInvocations, (int32_t)bm, (int32_t)bn, (int32_t)bk});
        }
      }
      return out;
    }

    void applyWinogradCoopmat2Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      applyTileFields(cfg, kWinogradCoopmat2TileFields, coopmat2TileScalars(t));
    }

    void applyWinogradCoopmat2AccF16Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      applyTileFields(cfg, kWinogradCoopmat2AccF16TileFields, coopmat2TileScalars(t));
    }

    // ---- Shared NHWC strided-GEMM / implicit-GEMM conv tuning Ops ----
    //
    // CoopmatOps<Traits> and Dot2Ops<Traits> unify the six Ops structs the
    // strided-GEMM and conv sweeps consume. Each Traits carries only what differs
    // (kernels, labels, tuned fields/bits, convEligible); the template provides the
    // shared sweep interface. Under the "tune everything" policy, trim/trimConv are
    // pure isConfigSupported filters -- no candidate ranking. The F16-coopmat variants
    // previously skipped the strided filter; bench() self-guards with the same predicate
    // anyway, so it only wasted sweep probes -- unify on always filtering. dot2 has no
    // conv family (no dot2 convolution kernel exists), so Dot2Ops omits the conv group.

    // Per-tile-shape formatting / equality / support dispatch used by the sweeps.
    std::string tileToString(const Dot2Tile& t) {
      return dot2TileToString(t);
    }
    std::string tileToString(const CoopmatTile& t) {
      return coopmatTileToString(t);
    }
    std::string tileToString(const Coopmat2Tile& t) {
      return coopmat2TileToString(t);
    }
    bool tilesEqual(const Dot2Tile& a, const Dot2Tile& b) {
      return dot2TilesEqual(a, b);
    }
    bool tilesEqual(const CoopmatTile& a, const CoopmatTile& b) {
      return coopmatTilesEqual(a, b);
    }
    bool tilesEqual(const Coopmat2Tile& a, const Coopmat2Tile& b) {
      return coopmat2TilesEqual(a, b);
    }

    template<typename Kernel>
    bool kernelSupportsTile(const CoopmatTile& t) {
      return Kernel::isConfigSupported(
        t.workgroupSize, t.bm, t.bn, t.bk, t.sgm, t.sgn, t.tm, t.tn, t.tk, t.subgroupSize);
    }
    template<typename Kernel>
    bool kernelSupportsTile(const Coopmat2Tile& t) {
      return Kernel::isConfigSupported(t.workgroupSize, t.bm, t.bn, t.bk);
    }
    template<typename Kernel>
    bool kernelSupportsTile(const Dot2Tile& t) {
      return Kernel::isConfigSupported(
        t.workgroupSize, t.bm, t.bn, t.sgm, t.sgn, t.sgmIter, t.tm, t.tn, t.subgroupSize);
    }

    // Shared strided-GEMM eligibility checks, the GEMM-side counterparts of the
    // conv checks below.
    //
    // isConfigSupported alone only enforces the kernel's *structural* rules
    // (subgroup-aligned tiling, divisibility). It takes no VkPhysicalDeviceLimits,
    // so on its own it happily admits tiles the device cannot actually run.
    // Compiling and dispatching such a tile is not a clean failure: on some
    // drivers it hangs the GPU long enough to trip the watchdog and the device is
    // lost for the rest of the process. Observed on a Tesla T4, where the
    // coopmat1 strided sweep reaches BM=32 BN=128 BK=128 -- 59904 bytes of shared
    // memory against a 49152-byte limit -- and takes the device down. So screen
    // the device limits here, before build().
    //
    // Shared-memory accounting per family:
    //   coopmat1: staged A + packed B + the per-subgroup C staging area, all from
    //             the kernel's own sharedBytes().
    //   coopmat2: the shaders declare no `shared` at all (operands stream through
    //             tensorLayoutNV / coopMatLoadTensorNV), so the only shared cost
    //             is the driver's workgroup-scope reservation.
    //   dot2:     the fixed [BM][BK+4] + [BN][BK+4] fp16 tile pair, BK being the
    //             shaders' compile-time `#define BK 32`.
    bool workgroupWithinLimits(const VkPhysicalDeviceLimits& limits, int32_t workgroupSize) {
      return (uint32_t)workgroupSize <= limits.maxComputeWorkGroupInvocations &&
             (uint32_t)workgroupSize <= limits.maxComputeWorkGroupSize[0];
    }

    template<typename GemmKernel>
    bool coopmat1GemmEligible(TuningContext& x, const CoopmatTile& t, const VkPhysicalDeviceLimits& limits) {
      return GemmKernel::isConfigSupported(
               t.workgroupSize, t.bm, t.bn, t.bk, t.sgm, t.sgn, t.tm, t.tn, t.tk, t.subgroupSize) &&
             t.subgroupSize == (int32_t)x.dev->info.subgroupSize && workgroupWithinLimits(limits, t.workgroupSize) &&
             GemmKernel::sharedBytes(t.workgroupSize, t.bm, t.bn, t.bk, t.tm, t.tn, t.subgroupSize) <=
               limits.maxComputeSharedMemorySize;
    }

    template<typename GemmKernel, std::vector<Coopmat2FlexShape> VulkanDeviceInfo::* FlexShapes>
    bool coopmat2GemmEligible(TuningContext& x, const Coopmat2Tile& t, const VkPhysicalDeviceLimits& limits) {
      // No shared-memory term here: these shaders declare no `shared` arrays, and
      // the driver's workgroup-scope reservation is a device constant that device
      // init already refuses to exceed (it gates supportsCoopmat2F16 itself), so
      // there is nothing tile-dependent left to check.
      return GemmKernel::isConfigSupported(t.workgroupSize, t.bm, t.bn, t.bk) &&
             coopmat2TileSupported(x.dev->info, x.dev->info.*FlexShapes, t) &&
             workgroupWithinLimits(limits, t.workgroupSize);
    }

    template<typename GemmKernel>
    bool dot2GemmEligible(TuningContext& x, const Dot2Tile& t, const VkPhysicalDeviceLimits& limits) {
      return GemmKernel::isConfigSupported(
               t.workgroupSize, t.bm, t.bn, t.sgm, t.sgn, t.sgmIter, t.tm, t.tn, t.subgroupSize) &&
             t.subgroupSize == (int32_t)x.dev->info.subgroupSize && workgroupWithinLimits(limits, t.workgroupSize) &&
             GemmKernel::sharedBytes(t.bm, t.bn) <= limits.maxComputeSharedMemorySize;
    }

    // Shared implicit-GEMM conv eligibility checks. coopmat1 checks the classic
    // isConfigSupported (subgroup-aligned tiling) plus shared-memory and workgroup
    // limits. coopmat2 must additionally confirm the device reports the flex shape --
    // the workgroup-scope fragment IS the BM x BN tile, so a tile the device does not
    // report silently produces wrong results rather than failing -- and its shared A
    // tile (sharedBytes) adds to the coopmat reserved shared budget.
    template<typename ConvKernel>
    bool
    coopmat1ConvEligible(TuningContext& x, const CoopmatTile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
      return ConvKernel::isConfigSupported(
               t.workgroupSize, t.bm, t.bn, t.bk, t.sgm, t.sgn, t.tm, t.tn, t.tk, t.subgroupSize) &&
             t.subgroupSize == (int32_t)x.dev->info.subgroupSize &&
             (uint32_t)t.workgroupSize <= limits.maxComputeWorkGroupInvocations &&
             (uint32_t)t.workgroupSize <= limits.maxComputeWorkGroupSize[0] && spatial % t.bm == 0 &&
             ConvKernel::sharedBytes(t.bm, t.bn, t.bk) <= limits.maxComputeSharedMemorySize;
    }

    template<typename ConvKernel, std::vector<Coopmat2FlexShape> VulkanDeviceInfo::* FlexShapes>
    bool
    coopmat2ConvEligible(TuningContext& x, const Coopmat2Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
      const size_t userShared = ConvKernel::sharedBytes(t.bm, t.bk);
      const size_t totalShared = (size_t)x.dev->info.coopmat2ReservedSharedBytes + userShared;
      return ConvKernel::isConfigSupported(t.workgroupSize, t.bm, t.bn, t.bk) &&
             coopmat2TileSupported(x.dev->info, x.dev->info.*FlexShapes, t) &&
             totalShared <= limits.maxComputeSharedMemorySize &&
             (uint32_t)t.workgroupSize <= limits.maxComputeWorkGroupInvocations &&
             (uint32_t)t.workgroupSize <= limits.maxComputeWorkGroupSize[0] && spatial % t.bm == 0;
    }

    template<typename Traits>
    struct CoopmatOps {
      using Tile = typename Traits::Tile;
      static string_view gemmLabel() { return Traits::gemmLabel(); }
      static int bm(const Tile& t) { return t.bm; }
      static int bn(const Tile& t) { return t.bn; }
      static std::string format(const Tile& t) { return tileToString(t); }
      static bool equal(const Tile& a, const Tile& b) { return tilesEqual(a, b); }
      static std::vector<Tile> trim(const std::vector<Tile>& c, int, int, int) {
        std::vector<Tile> trimmed;
        for(const Tile& t: c)
          if(kernelSupportsTile<typename Traits::GemmKernel>(t))
            trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c, const Tile& t) { Traits::applyGemm(c, t); }
      static void saveGemm(VulkanTuneParams& c, const Tile& t) {
        applyGemm(c, t);
        c.markKernelTuned({Traits::gemmTunedBit});
      }
      static int32_t& timeUs(VulkanTuneParams& c) { return c.*Traits::timeUsField; }
      static bool splitOutputModes() { return true; }
      static KernelBench
      benchGemm(TuningContext& x, const VulkanTuneParams& c, int it, int b, int m, int n, int k, bool addToOutput) {
        return Traits::GemmKernel::bench(x, c, it, b, m, n, k, addToOutput);
      }
      static string_view convLabel(int convSize) { return Traits::convLabel(convSize); }
      static std::vector<Tile> trimConv(const std::vector<Tile>& c, int, int, int) {
        std::vector<Tile> trimmed;
        for(const Tile& t: c)
          if(kernelSupportsTile<typename Traits::ConvKernel>(t))
            trimmed.push_back(t);
        return trimmed;
      }
      static bool convEligible(TuningContext& x, const Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
        return Traits::convEligible(x, t, spatial, limits);
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return Traits::gemmEligible(x, t, limits);
      }
      static void applyConv(VulkanTuneParams& c, const Tile& t, int convSize) { Traits::applyConv(c, t, convSize); }
      static void saveConv(VulkanTuneParams& c, const Tile& t, int convSize) { applyConv(c, t, convSize); }
      static void markConvValid(VulkanTuneParams& c, int convSize) {
        if(convSize == 5)
          c.markKernelTuned({Traits::conv5TunedBit});
        else
          c.markKernelTuned({Traits::conv3TunedBit});
      }
      static KernelBench benchConv(
        TuningContext& x,
        const VulkanTuneParams& c,
        int it,
        int b,
        int nx,
        int ny,
        int s,
        int convSize,
        int in,
        int out) {
        return Traits::ConvKernel::bench(x, c, it, b, nx, ny, s, convSize, in, out);
      }
      static KernelBench benchConvReference(
        TuningContext& x,
        const VulkanTuneParams& c,
        const std::vector<Tile>& tiles,
        int it,
        int b,
        int nx,
        int ny,
        int s,
        int convSize,
        int in,
        int out) {
        // coopmat2 variants prefer a coopmat1 reference (cheaper to run) when the
        // device can build one; coopmat1 has no such fallback.
        if constexpr(Traits::hasReferenceFallback) {
          if(Traits::referenceFallbackSupported(x)) {
            KernelBench ref = Traits::ReferenceFallbackKernel::bench(x, c, it, b, nx, ny, s, convSize, in, out);
            if(ref.ok)
              return ref;
          }
        }
        for(const Tile& tile: tiles) {
          VulkanTuneParams trial = c;
          applyConv(trial, tile, convSize);
          KernelBench ref = benchConv(x, trial, it, b, nx, ny, s, convSize, in, out);
          if(ref.ok)
            return ref;
        }
        return KernelBench();
      }
    };

    struct NhwcCoopmat1F32Traits {
      using Tile = CoopmatTile;
      using GemmKernel = VulkanKernels::GemmStridedCoopmat1Nhwc;
      using ConvKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8;
      static string_view gemmLabel() { return "gemmStridedCoopmat1AccF32Nhwc"; }
      static string_view convLabel(int convSize) { return convSize == 5 ? "conv5x5NhwcCoopmat" : "conv3x3NhwcCoopmat"; }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_COOPMAT1;
      static constexpr int64_t conv3TunedBit = TUNED_CONV3X3_COOPMAT1;
      static constexpr int64_t conv5TunedBit = TUNED_CONV5X5_COOPMAT1;
      static constexpr int32_t VulkanTuneParams::* conv3TimeUsField = &VulkanTuneParams::conv3x3NhwcCoopmat1TimeUs;
      static constexpr int32_t VulkanTuneParams::* conv5TimeUsField = &VulkanTuneParams::conv5x5NhwcCoopmat1TimeUs;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmCoopmat1F32TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedCoopmat1TileFields, coopmatTileScalars(t));
      }
      static void applyConv(VulkanTuneParams& c, const Tile& t, int convSize) {
        applyTileFields(
          c, convSize == 5 ? kConv5x5Coopmat1TileFields : kConv3x3Coopmat1TileFields, coopmatTileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return coopmat1GemmEligible<GemmKernel>(x, t, limits);
      }
      static bool convEligible(TuningContext& x, const Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
        return coopmat1ConvEligible<ConvKernel>(x, t, spatial, limits);
      }
      static constexpr bool hasReferenceFallback = false;
    };
    using NhwcCoopmat1F32Ops = CoopmatOps<NhwcCoopmat1F32Traits>;

    struct NhwcCoopmat2F32Traits {
      using Tile = Coopmat2Tile;
      using GemmKernel = VulkanKernels::GemmStridedCoopmat2Nhwc;
      using ConvKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8;
      using ReferenceFallbackKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8;
      static string_view gemmLabel() { return "gemmStridedCoopmat2AccF32Nhwc"; }
      static string_view convLabel(int convSize) {
        return convSize == 5 ? "conv5x5NhwcCoopmat2" : "conv3x3NhwcCoopmat2";
      }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_COOPMAT2;
      static constexpr int64_t conv3TunedBit = TUNED_CONV3X3_COOPMAT2;
      static constexpr int64_t conv5TunedBit = TUNED_CONV5X5_COOPMAT2;
      static constexpr int32_t VulkanTuneParams::* conv3TimeUsField = &VulkanTuneParams::conv3x3NhwcCoopmat2TimeUs;
      static constexpr int32_t VulkanTuneParams::* conv5TimeUsField = &VulkanTuneParams::conv5x5NhwcCoopmat2TimeUs;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmCoopmat2F32TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedCoopmat2TileFields, coopmat2TileScalars(t));
      }
      static void applyConv(VulkanTuneParams& c, const Tile& t, int convSize) {
        applyTileFields(
          c, convSize == 5 ? kConv5x5Coopmat2TileFields : kConv3x3Coopmat2TileFields, coopmat2TileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return coopmat2GemmEligible<GemmKernel, &VulkanDeviceInfo::coopmat2FlexShapes>(x, t, limits);
      }
      static bool convEligible(TuningContext& x, const Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
        return coopmat2ConvEligible<ConvKernel, &VulkanDeviceInfo::coopmat2FlexShapes>(x, t, spatial, limits);
      }
      static constexpr bool hasReferenceFallback = true;
      static bool referenceFallbackSupported(TuningContext& x) { return x.dev->info.supportsCoopmat1F16; }
    };
    using NhwcCoopmat2F32Ops = CoopmatOps<NhwcCoopmat2F32Traits>;

    struct NhwcCoopmat1Traits {
      using Tile = CoopmatTile;
      using GemmKernel = VulkanKernels::GemmStridedCoopmat1AccF16Nhwc;
      using ConvKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8;
      static string_view gemmLabel() { return "gemmStridedCoopmat1AccF16Nhwc"; }
      static string_view convLabel(int convSize) {
        return convSize == 5 ? "conv5x5NhwcCoopmatAccF16" : "conv3x3NhwcCoopmatAccF16";
      }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_COOPMAT1_ACCF16;
      static constexpr int64_t conv3TunedBit = TUNED_CONV3X3_COOPMAT1_ACCF16;
      static constexpr int64_t conv5TunedBit = TUNED_CONV5X5_COOPMAT1_ACCF16;
      static constexpr int32_t VulkanTuneParams::* conv3TimeUsField = &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16TimeUs;
      static constexpr int32_t VulkanTuneParams::* conv5TimeUsField = &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16TimeUs;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmCoopmat1F16TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedCoopmat1AccF16TileFields, coopmatTileScalars(t));
      }
      static void applyConv(VulkanTuneParams& c, const Tile& t, int convSize) {
        applyTileFields(
          c,
          convSize == 5 ? kConv5x5Coopmat1AccF16TileFields : kConv3x3Coopmat1AccF16TileFields,
          coopmatTileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return coopmat1GemmEligible<GemmKernel>(x, t, limits);
      }
      static bool convEligible(TuningContext& x, const Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
        return coopmat1ConvEligible<ConvKernel>(x, t, spatial, limits);
      }
      static constexpr bool hasReferenceFallback = false;
    };
    using NhwcCoopmat1Ops = CoopmatOps<NhwcCoopmat1Traits>;

    struct NhwcCoopmat2Traits {
      using Tile = Coopmat2Tile;
      using GemmKernel = VulkanKernels::GemmStridedCoopmat2AccF16Nhwc;
      using ConvKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8;
      using ReferenceFallbackKernel = VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8;
      static string_view gemmLabel() { return "gemmStridedCoopmat2AccF16Nhwc"; }
      static string_view convLabel(int convSize) {
        return convSize == 5 ? "conv5x5NhwcCoopmat2AccF16" : "conv3x3NhwcCoopmat2AccF16";
      }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_COOPMAT2_ACCF16;
      static constexpr int64_t conv3TunedBit = TUNED_CONV3X3_COOPMAT2_ACCF16;
      static constexpr int64_t conv5TunedBit = TUNED_CONV5X5_COOPMAT2_ACCF16;
      static constexpr int32_t VulkanTuneParams::* conv3TimeUsField = &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16TimeUs;
      static constexpr int32_t VulkanTuneParams::* conv5TimeUsField = &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16TimeUs;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmCoopmat2F16TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedCoopmat2AccF16TileFields, coopmat2TileScalars(t));
      }
      static void applyConv(VulkanTuneParams& c, const Tile& t, int convSize) {
        applyTileFields(
          c,
          convSize == 5 ? kConv5x5Coopmat2AccF16TileFields : kConv3x3Coopmat2AccF16TileFields,
          coopmat2TileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return coopmat2GemmEligible<GemmKernel, &VulkanDeviceInfo::coopmat2AccF16FlexShapes>(x, t, limits);
      }
      static bool convEligible(TuningContext& x, const Tile& t, int spatial, const VkPhysicalDeviceLimits& limits) {
        // BK's shared-memory cost: A stages through a [BM][BK+8] shared tile, so larger
        // BK now costs shared memory -- but convEligible enforces the device limit, and
        // fewer, wider slabs still tend to win. Let the sweep judge up to 64.
        return coopmat2ConvEligible<ConvKernel, &VulkanDeviceInfo::coopmat2AccF16FlexShapes>(x, t, spatial, limits);
      }
      static constexpr bool hasReferenceFallback = true;
      static bool referenceFallbackSupported(TuningContext& x) { return x.dev->info.supportsCoopmat1F16AccF16; }
    };
    using NhwcCoopmat2Ops = CoopmatOps<NhwcCoopmat2Traits>;

    // dot2 has no conv family (no dot2 convolution kernel exists): tuneGemmStridedSweep
    // only calls gemmLabel/bm/bn/trim/applyGemm/saveGemm/benchGemm/format/equal/timeUs,
    // all provided here.
    template<typename Traits>
    struct Dot2Ops {
      using Tile = typename Traits::Tile;
      static string_view gemmLabel() { return Traits::gemmLabel(); }
      static int bm(const Tile& t) { return t.bm; }
      static int bn(const Tile& t) { return t.bn; }
      static std::string format(const Tile& t) { return tileToString(t); }
      static bool equal(const Tile& a, const Tile& b) { return tilesEqual(a, b); }
      static std::vector<Tile> trim(const std::vector<Tile>& c, int, int, int) {
        std::vector<Tile> trimmed;
        for(const Tile& t: c)
          if(kernelSupportsTile<typename Traits::GemmKernel>(t))
            trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c, const Tile& t) { Traits::applyGemm(c, t); }
      static void saveGemm(VulkanTuneParams& c, const Tile& t) {
        applyGemm(c, t);
        c.markKernelTuned({Traits::gemmTunedBit});
      }
      static int32_t& timeUs(VulkanTuneParams& c) { return c.*Traits::timeUsField; }
      static bool splitOutputModes() { return true; }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return Traits::gemmEligible(x, t, limits);
      }
      static KernelBench
      benchGemm(TuningContext& x, const VulkanTuneParams& c, int it, int b, int m, int n, int k, bool addToOutput) {
        return Traits::GemmKernel::bench(x, c, it, b, m, n, k, addToOutput);
      }
    };

    struct NhwcDot2F32Traits {
      using Tile = Dot2Tile;
      using GemmKernel = VulkanKernels::GemmStridedDot2Nhwc;
      static string_view gemmLabel() { return "gemmStridedDot2AccF32Nhwc"; }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_DOT2;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmDot2F32TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedDot2TileFields, dot2TileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return dot2GemmEligible<GemmKernel>(x, t, limits);
      }
    };
    using NhwcDot2Ops = Dot2Ops<NhwcDot2F32Traits>;

    struct NhwcDot2AccF16Traits {
      using Tile = Dot2Tile;
      using GemmKernel = VulkanKernels::GemmStridedDot2AccF16Nhwc;
      static string_view gemmLabel() { return "gemmStridedDot2AccF16Nhwc"; }
      static constexpr int64_t gemmTunedBit = TUNED_GEMM_STRIDED_DOT2_ACCF16;
      static constexpr int32_t VulkanTuneParams::* timeUsField = &VulkanTuneParams::nhwcGemmDot2F16TimeUs;
      static void applyGemm(VulkanTuneParams& c, const Tile& t) {
        applyTileFields(c, kNhwcStridedDot2AccF16TileFields, dot2TileScalars(t));
      }
      static bool gemmEligible(TuningContext& x, const Tile& t, const VkPhysicalDeviceLimits& limits) {
        return dot2GemmEligible<GemmKernel>(x, t, limits);
      }
    };
    using NhwcDot2AccF16Ops = Dot2Ops<NhwcDot2AccF16Traits>;

    template<typename Ops>
    void tuneGemmStridedSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<typename Ops::Tile>& candidates,
      std::ostream& out,
      bool verbose,
      VulkanTuneParams& cfg) {
      using Tile = typename Ops::Tile;
      std::vector<GemmStridedModeCase> cases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(cases.empty() || candidates.empty())
        return;
      std::vector<Tile> sweep = Ops::trim(candidates, cases[0].M, cases[0].N, cases[0].K);
      // Drop tiles the device cannot run before any of them reaches build().
      // Ops::trim only applies the kernel's structural isConfigSupported; a tile
      // that overruns shared memory or the workgroup-invocation cap can hang the
      // GPU rather than fail cleanly, taking the whole tuning session with it.
      {
        const VkPhysicalDeviceLimits& limits = ctx.dev->info.properties.limits;
        std::vector<Tile> runnable;
        runnable.reserve(sweep.size());
        for(const Tile& t: sweep)
          if(Ops::gemmEligible(ctx, t, limits))
            runnable.push_back(t);
        if(runnable.size() != sweep.size())
          out << "VulkanTuner: " << Ops::gemmLabel() << " skipping " << (sweep.size() - runnable.size()) << " of "
              << sweep.size() << " candidate tiles that exceed device limits" << endl;
        sweep = std::move(runnable);
      }
      if(sweep.empty())
        return;
      struct CachedRef {
        int m, n, k;
        KernelBench bench;
      };
      std::vector<CachedRef> refs;
      auto getRef = [&](int m, int n, int k) -> const KernelBench* {
        for(const CachedRef& ref: refs)
          if(ref.m == m && ref.n == n && ref.k == k)
            return &ref.bench;
        VulkanKernels::GemmStridedTiledNhwc probe(problemBatchSize, m, n, k);
        refs.push_back({m, n, k, probe.bench(ctx, cfg, iters)});
        return &refs.back().bench;
      };
      auto bench = [&](const Tile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = cfg;
        Ops::applyGemm(trial, t);
        metrics = {};
        double flops = 0.0, seconds = 0.0;
        for(const auto& cs: cases) {
          const int n = roundUpCoopmatDim(cs.N, Ops::bn(t));
          const KernelBench* ref = getRef(cs.M, n, cs.K);
          if(ref == nullptr || !ref->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          const int modeOccurrences[2] = {
            Ops::splitOutputModes() ? cs.overwriteOccurrences : cs.occurrences,
            Ops::splitOutputModes() ? cs.addToOutputOccurrences : 0};
          for(int mode = 0; mode < 2; mode++) {
            const int occurrences = modeOccurrences[mode];
            if(occurrences <= 0)
              continue;
            KernelBench b = Ops::benchGemm(ctx, trial, iters, problemBatchSize, cs.M, n, cs.K, mode != 0);
            if(!(b.ok && b.kernelsPerSecond > 0.0))
              return rejectAggregateCandidate(metrics, "bench_failed");
            double rmse = normalizedRmse(ref->output, b.output);
            metrics.maxRmse = std::max(metrics.maxRmse, rmse);
            if(rmse > MAX_RMSE_TOLERANCE)
              return rejectAggregateCandidate(metrics, "rmse_exceeded");
            double wt = dispatchWeightedTime(b, occurrences);
            metrics.weightedTime += wt;
            seconds += wt;
            flops += 2.0 * problemBatchSize * cs.M * n * cs.K * occurrences;
          }
        }
        metrics.tflops = aggregateTflops(flops, seconds);
        return metrics.weightedTime > 0.0;
      };
      const string_view label = Ops::gemmLabel();
      auto result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: strided GEMM found no valid tile."},
        sweep,
        TUNER_CONFIRM_K,
        true,
        true,
        true,
        out,
        verbose,
        ctx,
        bench,
        Ops::format,
        Ops::equal);
      if(result.found) {
        Ops::saveGemm(cfg, result.candidate);
        ModeSelectStats stats{true, result.weightedTime, 0.0};
        Ops::timeUs(cfg) = weightedTimeToMicros(stats);
      }
    }

    // ---- Shared winograd GEMM tile-sweep machinery ----

    // Per-family knobs for the shared winograd GEMM tile sweep below.
    struct WinogradDot2Family {
      using Tile = Dot2Tile;
      using KernelF32 = VulkanKernels::WinogradGemmDot2;
      using KernelF16 = VulkanKernels::WinogradGemmDot2AccF16;
      static string_view label(bool accF16) { return accF16 ? "winogradDot2AccF16" : "winogradDot2"; }
      static void apply(VulkanTuneParams& cfg, const Tile& t, bool accF16) {
        if(accF16)
          applyWinogradDot2AccF16Tile(cfg, t);
        else
          applyWinogradDot2Tile(cfg, t);
      }
      static int64_t tunedBit(bool accF16) { return accF16 ? TUNED_WINOGRAD_DOT2_ACCF16 : TUNED_WINOGRAD_DOT2; }
    };
    struct WinogradCoopmat1Family {
      using Tile = CoopmatTile;
      using KernelF32 = VulkanKernels::WinogradGemmCoopmat1;
      using KernelF16 = VulkanKernels::WinogradGemmCoopmat1AccF16;
      static string_view label(bool accF16) { return accF16 ? "winogradCoopmat1AccF16" : "winogradCoopmat1"; }
      static void apply(VulkanTuneParams& cfg, const Tile& t, bool accF16) {
        if(accF16)
          applyWinogradCoopmatAccF16Tile(cfg, t);
        else
          applyWinogradCoopmatTile(cfg, t);
      }
      static int64_t tunedBit(bool accF16) { return accF16 ? TUNED_WINOGRAD_COOPMAT1_ACCF16 : TUNED_WINOGRAD_COOPMAT1; }
    };
    struct WinogradCoopmat2Family {
      using Tile = Coopmat2Tile;
      using KernelF32 = VulkanKernels::WinogradGemmCoopmat2;
      using KernelF16 = VulkanKernels::WinogradGemmCoopmat2AccF16;
      static string_view label(bool accF16) { return accF16 ? "winogradCoopmat2AccF16" : "winogradCoopmat2"; }
      static void apply(VulkanTuneParams& cfg, const Tile& t, bool accF16) {
        if(accF16)
          applyWinogradCoopmat2AccF16Tile(cfg, t);
        else
          applyWinogradCoopmat2Tile(cfg, t);
      }
      static int64_t tunedBit(bool accF16) { return accF16 ? TUNED_WINOGRAD_COOPMAT2_ACCF16 : TUNED_WINOGRAD_COOPMAT2; }
    };

    // Sweep one winograd GEMM family's tiles over the model's winograd mode cases,
    // selecting the fastest per-family tile for both accumulation modes.
    template<typename Family>
    void tuneWinogradTileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<typename Family::Tile>& candidates,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      using Tile = typename Family::Tile;
      using KernelF32 = typename Family::KernelF32;
      using KernelF16 = typename Family::KernelF16;
      std::vector<WinogradModeCase> sweepCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(sweepCases.empty() || candidates.empty())
        return;
      // The sweep benchmarks every device-valid candidate; the only pruning is dropping
      // tiles the kernel does not support (isConfigSupported), then deduping.
      std::vector<Tile> sweepCandidates;
      for(const Tile& t: candidates) {
        if(!kernelSupportsTile<KernelF32>(t))
          continue;
        bool duplicate = false;
        for(const Tile& existing: sweepCandidates)
          if(tilesEqual(existing, t)) {
            duplicate = true;
            break;
          }
        if(!duplicate)
          sweepCandidates.push_back(t);
      }
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

      auto benchAggregateTile = [&](const Tile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        Family::apply(trial, t, accF16);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const WinogradModeCase& cs: sweepCases) {
          const auto pad = accF16 ? padDims(cs.M, cs.N, cs.K, KernelF16::layerPaddingContract(trial))
                                  : padDims(cs.M, cs.N, cs.K, KernelF32::layerPaddingContract(trial));
          KernelBench bench;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            KernelF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            bench = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            KernelF32 probe(pad.m, pad.n, pad.k, cs.numBatches);
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
          if(rmse > MAX_RMSE_TOLERANCE)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");
          const double caseWeightedTime = dispatchWeightedTime(bench, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += flopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      const string_view label = Family::label(accF16);
      out << "VulkanTuner: " << label << " tile sweep candidates=" << sweepCandidates.size()
          << " cases=" << sweepCases.size() << endl;
      AggregateCandidateSweepResult<Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_K,
        true,
        true,
        true,
        out,
        verboseTuner,
        ctx,
        benchAggregateTile,
        [](const Tile& t) { return tileToString(t); },
        [](const Tile& a, const Tile& b) { return tilesEqual(a, b); });
      if(result.found) {
        Family::apply(tunedConfig, result.candidate, accF16);
        tunedConfig.markKernelTuned({Family::tunedBit(accF16)});
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
      bool full,
      VulkanTuneParams& tunedConfig) {
      if(ctx.dev == nullptr || modelDesc == nullptr)
        return;
      const bool supportsFP16Compute = ctx.fp16Compute;
      const int problemBatchSize = std::max(1, batchSize);
      const bool fp16Ready = ctx.fp16Storage && supportsFP16Compute;
      const int64_t dm = ctx.dev->info.disabledAccelVariantMask;
      const bool coop1 = ctx.dev->info.supportsCoopmat1F16 && (dm & coopmat1AccF32Bits()) == 0;
      const bool coop1f16 = ctx.dev->info.supportsCoopmat1F16AccF16 && (dm & coopmat1AccF16Bits()) == 0;
      const bool coop2 = ctx.dev->info.supportsCoopmat2F16 && (dm & coopmat2AccF32Bits()) == 0;
      const bool coop2f16 = ctx.dev->info.supportsCoopmat2F16AccF16 && (dm & coopmat2AccF16Bits()) == 0;
      const bool dot2v = ctx.dev->info.supportsDot2F16 && (dm & dot2AccF32Bits()) == 0;
      const bool dot2f16 = ctx.dev->info.supportsDot2F16AccF16 && (dm & dot2AccF16Bits()) == 0;

      const bool supportsNhwcDot2 = fp16Ready && (dot2v || dot2f16);
      if(
        coop1 && ctx.fp16Storage && supportsFP16Compute &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1))) {
        tunedConfig.nhwcGemmCoopmat1F32TimeUs = 0;
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
        tuneGemmStridedSweep<NhwcCoopmat1F32Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      if(
        coop1f16 && ctx.fp16Storage && supportsFP16Compute &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1_ACCF16))) {
        tunedConfig.nhwcGemmCoopmat1F16TimeUs = 0;
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
        tuneGemmStridedSweep<NhwcCoopmat1Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      if(
        coop2 && ctx.fp16Storage && supportsFP16Compute &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2))) {
        tunedConfig.nhwcGemmCoopmat2F32TimeUs = 0;
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2FlexShapes, Coopmat2CandidateFamily::StridedGemm);
        tuneGemmStridedSweep<NhwcCoopmat2F32Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      if(
        coop2f16 && ctx.fp16Storage && supportsFP16Compute &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2_ACCF16))) {
        tunedConfig.nhwcGemmCoopmat2F16TimeUs = 0;
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2AccF16FlexShapes, Coopmat2CandidateFamily::StridedGemm);
        tuneGemmStridedSweep<NhwcCoopmat2Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      const bool haveNhwcCoop = (coop1 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1)) ||
                                (coop1f16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1_ACCF16)) ||
                                (coop2 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2)) ||
                                (coop2f16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2_ACCF16));
      if(
        (full || !haveNhwcCoop) && dot2v && fp16Ready &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2))) {
        tunedConfig.nhwcGemmDot2F32TimeUs = 0;
        std::vector<Dot2Tile> candidates = makeDot2Candidates(ctx);
        tuneGemmStridedSweep<NhwcDot2Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      if(
        (full || !haveNhwcCoop) && dot2f16 && fp16Ready &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2_ACCF16))) {
        tunedConfig.nhwcGemmDot2F16TimeUs = 0;
        std::vector<Dot2Tile> candidates = makeDot2Candidates(ctx);
        tuneGemmStridedSweep<NhwcDot2AccF16Ops>(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, out, verboseTuner, tunedConfig);
      }
      const bool haveNhwcDot2 = (dot2v && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2)) ||
                                (dot2f16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2_ACCF16));
      // The tiled NHWC strided GEMM is the fallback tier: sweep it when no
      // accelerated variant exists or none produced a valid winner (always in
      // full-tune mode). This unifies the no-accelerant and lost-accelerant
      // cases, which previously swept tiled twice in full mode.
      if(
        (full || (!haveNhwcCoop && !haveNhwcDot2)) &&
        (full || !tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_TILED)) &&
        !collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen).empty()) {
        out << endl << "VulkanTuner: tuning gemmStridedTiledNhwc..." << endl;
        VulkanTuneParams candidate = tunedConfig;
        if(
          tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiledNhwc>(
            "gemmStridedTiledNhwc",
            modelDesc,
            problemBatchSize,
            nnXLen,
            nnYLen,
            tunedConfig,
            ctx,
            iters,
            out,
            verboseTuner,
            candidate))
          tunedConfig = candidate;
      }
      bool implicit3Selected = false;
      bool implicit5Selected = false;
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV3X3_COOPMAT1)) && coop1 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
        implicit3Selected =
          tuneNhwcConvSweep<NhwcCoopmat1F32Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 3, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit3Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV3X3_COOPMAT2)) && coop2 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2FlexShapes, Coopmat2CandidateFamily::ImplicitConv, 64);
        implicit3Selected =
          tuneNhwcConvSweep<NhwcCoopmat2F32Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 3, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit3Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV3X3_COOPMAT1_ACCF16)) && coop1f16 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
        implicit3Selected =
          tuneNhwcConvSweep<NhwcCoopmat1Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 3, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit3Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV3X3_COOPMAT2_ACCF16)) && coop2f16 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2AccF16FlexShapes, Coopmat2CandidateFamily::ImplicitConv, 64);
        implicit3Selected =
          tuneNhwcConvSweep<NhwcCoopmat2Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 3, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit3Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV5X5_COOPMAT1)) && coop1 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
        implicit5Selected =
          tuneNhwcConvSweep<NhwcCoopmat1F32Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 5, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit5Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV5X5_COOPMAT2)) && coop2 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2FlexShapes, Coopmat2CandidateFamily::ImplicitConv, 64);
        implicit5Selected =
          tuneNhwcConvSweep<NhwcCoopmat2F32Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 5, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit5Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV5X5_COOPMAT1_ACCF16)) && coop1f16 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
        implicit5Selected =
          tuneNhwcConvSweep<NhwcCoopmat1Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 5, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit5Selected;
      }
      if(
        (full || !tunedConfig.hasKernelTuned(TUNED_CONV5X5_COOPMAT2_ACCF16)) && coop2f16 && ctx.fp16Storage &&
        supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(
          ctx, ctx.dev->info.coopmat2AccF16FlexShapes, Coopmat2CandidateFamily::ImplicitConv, 64);
        implicit5Selected =
          tuneNhwcConvSweep<NhwcCoopmat2Ops>(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, 5, iters, candidates, out, verboseTuner, tunedConfig) ||
          implicit5Selected;
      }
      const bool needsImplicit3 = !collectConv3x3NhwcModeCases(modelDesc, 3).empty();
      const bool needsImplicit5 = !collectConv3x3NhwcModeCases(modelDesc, 5).empty();
      const bool haveImplicit3 = !needsImplicit3 || implicit3Selected;
      const bool haveImplicit5 = !needsImplicit5 || implicit5Selected;
      const bool implicitCoopAvailable = haveImplicit3 && haveImplicit5;
      if(full || !implicitCoopAvailable) {
        const bool needsWinograd =
          !collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig).empty();
        const bool supportsWinogradCoop = fp16Ready && (coop1 || coop1f16 || coop2 || coop2f16);
        if((full || needsWinograd) && supportsWinogradCoop) {
          if(coop1 && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1))) {
            tunedConfig.winogradGemmCoopmat1F32TimeUs = 0;
            auto candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
            tuneWinogradTileSweep<WinogradCoopmat1Family>(
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
          if(coop1f16 && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1_ACCF16))) {
            tunedConfig.winogradGemmCoopmat1F16TimeUs = 0;
            auto candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
            tuneWinogradTileSweep<WinogradCoopmat1Family>(
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
          if(coop2 && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2))) {
            tunedConfig.winogradGemmCoopmat2F32TimeUs = 0;
            auto candidates = makeCoopmat2Candidates(
              ctx, ctx.dev->info.coopmat2FlexShapes, Coopmat2CandidateFamily::Winograd);
            tuneWinogradTileSweep<WinogradCoopmat2Family>(
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
          if(coop2f16 && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2_ACCF16))) {
            tunedConfig.winogradGemmCoopmat2F16TimeUs = 0;
            auto candidates = makeCoopmat2Candidates(
              ctx, ctx.dev->info.coopmat2AccF16FlexShapes, Coopmat2CandidateFamily::Winograd);
            tuneWinogradTileSweep<WinogradCoopmat2Family>(
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
        const bool haveWinogradCoop = (coop1 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1)) ||
                                      (coop1f16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1_ACCF16)) ||
                                      (coop2 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2)) ||
                                      (coop2f16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2_ACCF16));
        if((full || (needsWinograd && !haveWinogradCoop)) && (full || supportsNhwcDot2)) {
          if(dot2v && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2))) {
            tunedConfig.winogradGemmDot2F32TimeUs = 0;
            tuneWinogradTileSweep<WinogradDot2Family>(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              makeDot2Candidates(ctx),
              false,
              out,
              verboseTuner,
              tunedConfig);
          }
          if(dot2f16 && (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2_ACCF16))) {
            tunedConfig.winogradGemmDot2F16TimeUs = 0;
            tuneWinogradTileSweep<WinogradDot2Family>(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              makeDot2Candidates(ctx),
              true,
              out,
              verboseTuner,
              tunedConfig);
          }
        }
        const bool haveWinogradDot2 = (dot2v && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2)) ||
                                      (dot2f16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2_ACCF16));
        if(
          (full || (needsWinograd && !haveWinogradCoop && !haveWinogradDot2)) &&
          (full || !tunedConfig.hasKernelTuned(TUNED_WINOGRAD_TILED))) {
          VulkanTuneParams candidate = tunedConfig;
          if(
            tuneWinogradGemmTiledMultiShape<VulkanKernels::WinogradGemm>(
              "winogradGemmTiled",
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              tunedConfig,
              ctx,
              iters,
              out,
              verboseTuner,
              candidate))
            tunedConfig = candidate;
        }
      }
      return;
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

      const int problemBatchSize = std::max(1, batchSize);
      auto selectWinogradTier = [&]() {
        const bool cm1 = fp16Ready && info->supportsCoopmat1F16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1);
        const bool cm1f16 =
          fp16Ready && info->supportsCoopmat1F16AccF16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1_ACCF16);
        const bool cm2 = fp16Ready && info->supportsCoopmat2F16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2);
        const bool cm2f16 =
          fp16Ready && info->supportsCoopmat2F16AccF16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2_ACCF16);
        if(cm1 || cm1f16 || cm2 || cm2f16) {
          const bool scoresComplete = (!cm1 || tunedConfig.winogradGemmCoopmat1F32TimeUs > 0) &&
                                      (!cm1f16 || tunedConfig.winogradGemmCoopmat1F16TimeUs > 0) &&
                                      (!cm2 || tunedConfig.winogradGemmCoopmat2F32TimeUs > 0) &&
                                      (!cm2f16 || tunedConfig.winogradGemmCoopmat2F16TimeUs > 0);
          if(!scoresComplete) {
            if(cm1)
              tuneWinogradGemmVariantModeSelect<
                VulkanKernels::WinogradGemmCoopmat1,
                &VulkanTuneParams::winogradGemmCoopmat1F32TimeUs>(
                ctx,
                modelDesc,
                problemBatchSize,
                nnXLen,
                nnYLen,
                iters,
                "winogradGemmCoopmat1",
                out,
                verboseTuner,
                tunedConfig);
            if(cm1f16)
              tuneWinogradGemmVariantModeSelect<
                VulkanKernels::WinogradGemmCoopmat1AccF16,
                &VulkanTuneParams::winogradGemmCoopmat1F16TimeUs>(
                ctx,
                modelDesc,
                problemBatchSize,
                nnXLen,
                nnYLen,
                iters,
                "winogradGemmCoopmat1AccF16",
                out,
                verboseTuner,
                tunedConfig);
            if(cm2)
              tuneWinogradGemmVariantModeSelect<
                VulkanKernels::WinogradGemmCoopmat2,
                &VulkanTuneParams::winogradGemmCoopmat2F32TimeUs>(
                ctx,
                modelDesc,
                problemBatchSize,
                nnXLen,
                nnYLen,
                iters,
                "winogradGemmCoopmat2",
                out,
                verboseTuner,
                tunedConfig);
            if(cm2f16)
              tuneWinogradGemmVariantModeSelect<
                VulkanKernels::WinogradGemmCoopmat2AccF16,
                &VulkanTuneParams::winogradGemmCoopmat2F16TimeUs>(
                ctx,
                modelDesc,
                problemBatchSize,
                nnXLen,
                nnYLen,
                iters,
                "winogradGemmCoopmat2AccF16",
                out,
                verboseTuner,
                tunedConfig);
          }
          return;
        }
        const bool dot = fp16Ready && info->supportsDot2F16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2);
        const bool dotf16 =
          fp16Ready && info->supportsDot2F16AccF16 && tunedConfig.hasKernelTuned(TUNED_WINOGRAD_DOT2_ACCF16);
        const bool scoresComplete =
          (!dot || tunedConfig.winogradGemmDot2F32TimeUs > 0) && (!dotf16 || tunedConfig.winogradGemmDot2F16TimeUs > 0);
        if(!scoresComplete) {
          if(dot)
            tuneWinogradGemmVariantModeSelect<
              VulkanKernels::WinogradGemmDot2,
              &VulkanTuneParams::winogradGemmDot2F32TimeUs>(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              "winogradGemmDot2",
              out,
              verboseTuner,
              tunedConfig);
          if(dotf16)
            tuneWinogradGemmVariantModeSelect<
              VulkanKernels::WinogradGemmDot2AccF16,
              &VulkanTuneParams::winogradGemmDot2F16TimeUs>(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              "winogradGemmDot2AccF16",
              out,
              verboseTuner,
              tunedConfig);
        }
      };

      const bool cm1 =
        fp16Ready && info->supportsCoopmat1F16AccF16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1_ACCF16);
      const bool cm2 =
        fp16Ready && info->supportsCoopmat2F16AccF16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2_ACCF16);
      const bool cm1f32 =
        fp16Ready && info->supportsCoopmat1F16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1);
      const bool cm2f32 =
        fp16Ready && info->supportsCoopmat2F16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2);
      // The tile sweep (tuneGemmStridedSweep, above) already wrote the aggregate
      // weighted time to each variant's *TimeUs field when its confirm phase
      // succeeded, so read those directly rather than re-measuring. There is no
      // path in the current tuner that leaves hasKernelTuned(bit) set with a
      // zero TimeUs on a live sweep -- reconcile clears the tuned bit when a
      // score is missing (see computeInvalidVariantBitsFromTimeUs), and the
      // confirm phase always writes a nonzero value when it finds a candidate.
      // If a stale tune file gets here with tuned-bit-set but TimeUs=0, the
      // tier selector below correctly treats the score as unmeasured.
      const int32_t cm1F32Score = cm1f32 ? tunedConfig.nhwcGemmCoopmat1F32TimeUs : int32_t(0);
      const int32_t cm2F32Score = cm2f32 ? tunedConfig.nhwcGemmCoopmat2F32TimeUs : int32_t(0);
      const int32_t cm1F16Score = cm1 ? tunedConfig.nhwcGemmCoopmat1F16TimeUs : int32_t(0);
      const int32_t cm2F16Score = cm2 ? tunedConfig.nhwcGemmCoopmat2F16TimeUs : int32_t(0);

      if(cm1f32 || cm2f32 || cm1 || cm2) {
        // Selection is derived at runtime from these scores; report the winner
        // via the same tier/margin helper the runtime uses so the logged choice
        // cannot disagree with what will actually be dispatched.
        const VariantChoice coopTier[] = {
          {TUNED_GEMM_STRIDED_COOPMAT2, cm2F32Score, cm2f32, VARIANT_SELECT_PENALTY_NONE},
          {TUNED_GEMM_STRIDED_COOPMAT1, cm1F32Score, cm1f32, VARIANT_SELECT_PENALTY_NONE},
          {TUNED_GEMM_STRIDED_COOPMAT2_ACCF16, cm2F16Score, cm2, VARIANT_SELECT_PENALTY_ACCF16},
          {TUNED_GEMM_STRIDED_COOPMAT1_ACCF16, cm1F16Score, cm1, VARIANT_SELECT_PENALTY_ACCF16},
        };
        const int64_t winner = selectWithinTier(coopTier);
        const bool selected2 =
          winner == TUNED_GEMM_STRIDED_COOPMAT2 || winner == TUNED_GEMM_STRIDED_COOPMAT2_ACCF16;
        const bool selectedF16 =
          winner == TUNED_GEMM_STRIDED_COOPMAT1_ACCF16 || winner == TUNED_GEMM_STRIDED_COOPMAT2_ACCF16;
        out << "VulkanTuner: NHWC 1x1 coopmat winner=" << (winner == 0 ? "none" : (selected2 ? "coopmat2" : "coopmat1"))
            << " accumulation=" << (selectedF16 ? "fp16" : "fp32") << " scores_us=[" << cm1F32Score << ","
            << cm2F32Score << "," << cm1F16Score << "," << cm2F16Score << "]" << endl;
      } else {
        const bool dot = fp16Ready && info->supportsDot2F16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2);
        const bool dotf16 =
          fp16Ready && info->supportsDot2F16AccF16 && tunedConfig.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2_ACCF16);
        // Same as the coopmat branch above: the tile sweep already wrote these
        // aggregate weighted times. No re-measurement needed here.
        const int32_t dotF32Score = dot ? tunedConfig.nhwcGemmDot2F32TimeUs : int32_t(0);
        const int32_t dotF16Score = dotf16 ? tunedConfig.nhwcGemmDot2F16TimeUs : int32_t(0);
        // Dot2 with FP32 accumulation is the default whenever dot2 is available;
        // the runtime derives the FP16-vs-FP32 choice from these scores via the
        // same penalized-time helper used here.
        const VariantChoice dot2Tier[] = {
          {TUNED_GEMM_STRIDED_DOT2, dotF32Score, dot, VARIANT_SELECT_PENALTY_NONE},
          {TUNED_GEMM_STRIDED_DOT2_ACCF16, dotF16Score, dotf16, VARIANT_SELECT_PENALTY_ACCF16},
        };
        const bool dotUseF16 = selectWithinTier(dot2Tier) == TUNED_GEMM_STRIDED_DOT2_ACCF16;
        out << "VulkanTuner: NHWC 1x1 dot2 accumulation=" << (dotUseF16 ? "fp16" : "fp32") << " scores_us=["
            << dotF32Score << "," << dotF16Score << "]" << endl;
      }

      // Time every available implicit-GEMM conv variant (coopmat1/coopmat2 x
      // accF32/accF16) independently, so the runtime can apply the same
      // tier + penalty policy as the other kernel families.
      //
      // Each variant is timed on its own and RMSE-checked against a reference.
      // The reference is the first variant that benched successfully (an accF32
      // one whenever available, since accF32 is the more accurate baseline);
      // a variant whose output diverges past MAX_RMSE_TOLERANCE gets no score
      // and is therefore never selectable. Unlike the previous version this no
      // longer requires *both* coopmat families to be present -- a device with
      // only one family still gets a measured time for it.
      auto selectConvMode = [&](int convSize) {
        struct ConvVariant {
          const char* label;
          bool available;
          bool accF16;
          int32_t VulkanTuneParams::* timeField;
          KernelBench (*bench)(
            TuningContext&, const VulkanTuneParams&, int, int, int, int, int, int, int, int);
        };
        auto benchCm1F32 = +[](
                             TuningContext& c, const VulkanTuneParams& cfg, int it, int bs, int nx, int ny, int sp,
                             int cs_, int ic, int oc) {
          return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(
            c, cfg, it, bs, nx, ny, sp, cs_, ic, oc);
        };
        auto benchCm2F32 = +[](
                             TuningContext& c, const VulkanTuneParams& cfg, int it, int bs, int nx, int ny, int sp,
                             int cs_, int ic, int oc) {
          return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::bench(
            c, cfg, it, bs, nx, ny, sp, cs_, ic, oc);
        };
        auto benchCm1F16 = +[](
                             TuningContext& c, const VulkanTuneParams& cfg, int it, int bs, int nx, int ny, int sp,
                             int cs_, int ic, int oc) {
          return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::bench(
            c, cfg, it, bs, nx, ny, sp, cs_, ic, oc);
        };
        auto benchCm2F16 = +[](
                             TuningContext& c, const VulkanTuneParams& cfg, int it, int bs, int nx, int ny, int sp,
                             int cs_, int ic, int oc) {
          return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::bench(
            c, cfg, it, bs, nx, ny, sp, cs_, ic, oc);
        };
        const bool is5 = convSize == 5;
        auto tunedFor = [&](int64_t b3, int64_t b5) { return tunedConfig.hasKernelTuned(is5 ? b5 : b3); };
        ConvVariant variants[] = {
          {"coopmat1",
           fp16Ready && info->supportsCoopmat1F16 && tunedFor(TUNED_CONV3X3_COOPMAT1, TUNED_CONV5X5_COOPMAT1),
           false,
           is5 ? NhwcCoopmat1F32Traits::conv5TimeUsField : NhwcCoopmat1F32Traits::conv3TimeUsField,
           benchCm1F32},
          {"coopmat2",
           fp16Ready && info->supportsCoopmat2F16 && tunedFor(TUNED_CONV3X3_COOPMAT2, TUNED_CONV5X5_COOPMAT2),
           false,
           is5 ? NhwcCoopmat2F32Traits::conv5TimeUsField : NhwcCoopmat2F32Traits::conv3TimeUsField,
           benchCm2F32},
          {"coopmat1AccF16",
           fp16Ready && info->supportsCoopmat1F16AccF16 &&
             tunedFor(TUNED_CONV3X3_COOPMAT1_ACCF16, TUNED_CONV5X5_COOPMAT1_ACCF16),
           true,
           is5 ? NhwcCoopmat1Traits::conv5TimeUsField : NhwcCoopmat1Traits::conv3TimeUsField,
           benchCm1F16},
          {"coopmat2AccF16",
           fp16Ready && info->supportsCoopmat2F16AccF16 &&
             tunedFor(TUNED_CONV3X3_COOPMAT2_ACCF16, TUNED_CONV5X5_COOPMAT2_ACCF16),
           true,
           is5 ? NhwcCoopmat2Traits::conv5TimeUsField : NhwcCoopmat2Traits::conv3TimeUsField,
           benchCm2F16},
        };
        constexpr size_t nVariants = std::size(variants);

        for(auto& v: variants)
          tunedConfig.*(v.timeField) = 0;
        bool anyAvailable = false;
        for(const auto& v: variants)
          anyAvailable = anyAvailable || v.available;
        if(!anyAvailable)
          return;

        auto convCases = collectConv3x3NhwcModeCases(modelDesc, convSize);
        const int spatial = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
        double total[nVariants] = {};
        bool ok[nVariants];
        for(size_t i = 0; i < nVariants; i++)
          ok[i] = variants[i].available;

        for(const auto& cs: convCases) {
          const int batches[2] = {1, std::max(1, batchSize)};
          const int count = batches[0] == batches[1] ? 1 : 2;
          for(int bi = 0; bi < count; bi++) {
            const int bs = batches[bi];
            const double w = (bs == 1 && batchSize > 1 ? 4.0 : 1.0) * cs.occurrences;
            // Reference output for this case: first successful accF32 variant,
            // else first successful variant of any kind.
            std::vector<float> reference;
            KernelBench results[nVariants];
            for(size_t i = 0; i < nVariants; i++) {
              if(!ok[i])
                continue;
              results[i] = variants[i].bench(
                ctx, tunedConfig, iters, bs, nnXLen, nnYLen, spatial, convSize, cs.inChannels, cs.outChannels);
              if(!(results[i].ok && results[i].kernelsPerSecond > 0.0)) {
                ok[i] = false;
                continue;
              }
              if(reference.empty() && !variants[i].accF16)
                reference = results[i].output;
            }
            if(reference.empty())
              for(size_t i = 0; i < nVariants; i++)
                if(ok[i] && !results[i].output.empty()) {
                  reference = results[i].output;
                  break;
                }
            for(size_t i = 0; i < nVariants; i++) {
              if(!ok[i])
                continue;
              if(reference.empty() || normalizedRmse(reference, results[i].output) > MAX_RMSE_TOLERANCE) {
                ok[i] = false;
                continue;
              }
              total[i] += w / results[i].kernelsPerSecond;
            }
          }
        }

        out << "VulkanTuner: NHWC " << convSize << "x" << convSize << " conv scores_us=[";
        for(size_t i = 0; i < nVariants; i++) {
          if(ok[i])
            tunedConfig.*(variants[i].timeField) = secondsToMicros(total[i]);
          out << (i ? "," : "") << variants[i].label << "=" << tunedConfig.*(variants[i].timeField);
        }
        out << "]" << endl;
      };
      selectConvMode(3);
      selectConvMode(5);
      selectWinogradTier();
      return;
    }

  }  // namespace

  static bool retuneNhwcWinogradTransformsForCurrentLayout(
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
    const std::string retuneReason =
      tunedConfig.nhwcWinogradTransformTunedLayout.empty()
        ? "no previous NHWC Winograd transform layout tune recorded"
        : (nhwcWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.dev->info, ctx.fp16Storage, ctx.fp16Compute)
             ? "retune requested for selected Winograd GEMM layout"
             : "selected Winograd GEMM layout changed");
    auto tuneKernel = [&](auto& kernel) {
      out << endl
          << "VulkanTuner: re-tuning " << kernel.name()
          << " for selected Winograd GEMM layout (reason: " << retuneReason << ")..." << endl;
      std::vector<TunedCandidate> top = tuneOne(kernel, tunedConfig, ctx, iters, out, verboseTuner);
      if(top.empty())
        return false;
      applyCandidate(top[0], tunedConfig);
      return true;
    };
    VulkanKernels::WinogradTransformNhwc transform(batchSize, wgChannels, nnXLen, nnYLen);
    VulkanKernels::WinogradUntransformNhwc untransform(batchSize, wgChannels, nnXLen, nnYLen);
    const bool tunedTransform = tuneKernel(transform);
    const bool tunedUntransform = tuneKernel(untransform);
    // The attempt settles the transforms (like the attention tiers, which record
    // their validity as soon as they are evaluated); the layout signature is only
    // stamped when both kernels actually tuned. A settled-but-unstamped pair is
    // left alone by reconcile and re-attempted only by a full tune.
    tunedConfig.markKernelTuned({TUNED_WINOGRAD_TRANSFORM, TUNED_WINOGRAD_UNTRANSFORM});
    if(tunedTransform && tunedUntransform)
      markNhwcWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.dev->info, ctx.fp16Storage, ctx.fp16Compute);
    return tunedTransform && tunedUntransform;
  }

  static bool tuneGemmVariantsForCurrentHardware(
    TuningContext& ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    bool full,
    VulkanTuneParams& tunedConfig) {
    if(ctx.dev == nullptr || modelDesc == nullptr)
      return false;
    const VulkanTuneParams before = tunedConfig;
    const int iters = std::max(1, benchIters);
    out << endl << "VulkanTuner: updating GEMM variant candidates for current hardware..." << endl;
    tuneMissingGemmVariantCandidatesForCurrentHardware(
      ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, full, tunedConfig);
    selectGemmVariantsForCurrentHardware(
      ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
    // The NHWC transform/untransform kernels feed the selected Winograd GEMM
    // variant, so their layout depends on the selection above. Re-tune them for
    // the finalized layout whenever that path is in use and the recorded layout
    // is stale or absent (a full tune re-benchmarks them regardless).
    const int64_t required = requiredKernelMask(modelDesc, ctx.dev->info, ctx.fp16Storage, ctx.fp16Compute, full);
    if(
      (required & (TUNED_CONV3X3_WINOGRAD | TUNED_CONV5X5_WINOGRAD)) != 0 &&
      (full ||
       !nhwcWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.dev->info, ctx.fp16Storage, ctx.fp16Compute)))
      retuneNhwcWinogradTransformsForCurrentLayout(
        ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
    return !(tunedConfig == before);
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
              problem.headDim = roundUpToMultipleInt(attn->qHeadDim, 8);
              problem.vHeadDim = roundUpToMultipleInt(attn->vHeadDim, 8);
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
      int32_t workgroupSize, blockQ, blockKV, tm, tn, tk, subgroupSize;
      int32_t directKV;
      int32_t kvChunkCount;
    };

    string attentionCandidateString(const AttentionCoopmatCandidate& c) {
      return Global::strprintf(
        "[bs=%d bq=%d bkv=%d tm=%d tn=%d tk=%d subgroup=%d kv=%s chunks=%d]",
        c.workgroupSize,
        c.blockQ,
        c.blockKV,
        c.tm,
        c.tn,
        c.tk,
        c.subgroupSize,
        c.directKV != 0 ? "direct" : "staged",
        c.kvChunkCount);
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
      tunedConfig.markKernelTuned({TUNED_ATTN_COOPMAT1});
      tunedConfig.attnNhwcCoopmat1DirectKV = 0;
      tunedConfig.clearKernelTuned({TUNED_ATTN_COOPMAT1_SPLITK});
      tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch = 0;
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsCoopmat1F16)
        return false;
      AttentionTuneProblem p = findAttentionTuneProblem(modelDesc);
      if(p.headDim <= 0)
        return false;
      const int validTokens = nnXLen * nnYLen;
      const int seqLen = roundUpToMultipleInt(validTokens, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);
      auto makeCoopBench = [&](int benchBatchSize) -> std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> {
        if(maintenance1)
          return std::make_unique<VulkanKernels::AttentionCoopmat1Maintenance1Bench>(
            benchBatchSize,
            seqLen,
            p.headDim,
            p.vHeadDim,
            p.numHeads,
            p.numKVHeads,
            p.useRope,
            p.learnableRope,
            validTokens);
        return std::make_unique<VulkanKernels::AttentionCoopmat1Bench>(
          benchBatchSize,
          seqLen,
          p.headDim,
          p.vHeadDim,
          p.numHeads,
          p.numKVHeads,
          p.useRope,
          p.learnableRope,
          validTokens);
      };
      std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> coop = makeCoopBench(batchSize);

      KernelBench scalarRef = scalar.bench(ctx, tunedConfig, iters);
      if(!scalarRef.ok || scalarRef.kernelsPerSecond <= 0.0) {
        out << "VulkanTuner: attention variant reference bench failed; keeping tiled attention." << endl;
        return false;
      }
      std::vector<AttentionCoopmatCandidate> candidates;
      const int32_t subgroupSize = (int32_t)ctx.dev->info.subgroupSize;
      for(const CoopmatShape& shape: ctx.dev->info.coopmatShapes) {
        int32_t tm = (int32_t)shape.m, tn = (int32_t)shape.n, tk = (int32_t)shape.k;
        if(tm <= 0 || tn <= 0 || tk <= 0 || subgroupSize <= 0)
          continue;
        // maintenance1 converts each TN-wide probability fragment straight to
        // the TK-wide P×V A operand, so unlike the portable spill/reload path
        // it deliberately supports only square N/K fragment widths.
        if(maintenance1 && tn != tk)
          continue;
        int32_t kvMultiple = tn / gcdPositive(tn, tk) * tk;
        // 32/64 cover the regular and split-K winners on both the small and
        // large-model sweeps. 16 and 128 only added slow edge tiles, while
        // two/four subgroups cover the 64/128-thread winner classes without
        // compiling the 256-thread alternatives.
        for(int targetKV: {32, 64}) {
          int32_t bkv = ((targetKV + kvMultiple - 1) / kvMultiple) * kvMultiple;
          if(bkv > 128)
            continue;
          for(int subgroupMul: {2, 4}) {
            int32_t workgroupSize = subgroupSize * subgroupMul;
            int32_t bq = workgroupSize;
            for(int32_t directKV: {0, 1}) {
              if(directKV != 0 && seqLen % bkv != 0)
                continue;
              AttentionCoopmatCandidate c{workgroupSize, bq, bkv, tm, tn, tk, subgroupSize, directKV, 1};
              VulkanTuneParams cfg = tunedConfig;
              cfg.attnNhwcCoopmat1WorkgroupSize = c.workgroupSize;
              cfg.attnNhwcCoopmat1BlockQ = c.blockQ;
              cfg.attnNhwcCoopmat1BlockKV = c.blockKV;
              cfg.attnNhwcCoopmat1TM = c.tm;
              cfg.attnNhwcCoopmat1TN = c.tn;
              cfg.attnNhwcCoopmat1TK = c.tk;
              cfg.attnNhwcCoopmat1SubgroupSize = c.subgroupSize;
              cfg.attnNhwcCoopmat1DirectKV = c.directKV;
              if(!coop->validate(cfg, ctx.dev->info.properties.limits))
                continue;
              bool duplicate = false;
              for(const auto& old: candidates)
                duplicate =
                  duplicate || (old.workgroupSize == c.workgroupSize && old.blockQ == c.blockQ &&
                                old.blockKV == c.blockKV && old.tm == c.tm && old.tn == c.tn && old.tk == c.tk &&
                                old.subgroupSize == c.subgroupSize && old.directKV == c.directKV);
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
      double screenedBestTime = std::numeric_limits<double>::infinity();
      out << "VulkanTuner: attention coopmat1 candidates=" << candidates.size() << endl;
      for(const AttentionCoopmatCandidate& c: candidates) {
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcCoopmat1WorkgroupSize = c.workgroupSize;
        cfg.attnNhwcCoopmat1BlockQ = c.blockQ;
        cfg.attnNhwcCoopmat1BlockKV = c.blockKV;
        cfg.attnNhwcCoopmat1TM = c.tm;
        cfg.attnNhwcCoopmat1TN = c.tn;
        cfg.attnNhwcCoopmat1TK = c.tk;
        cfg.attnNhwcCoopmat1SubgroupSize = c.subgroupSize;
        cfg.attnNhwcCoopmat1DirectKV = c.directKV;
        KernelBench b = coop->bench(ctx, cfg, c.kvChunkCount, iters);
        double rmse = b.ok ? normalizedRmse(scalarRef.output, b.output) : std::numeric_limits<double>::infinity();
        bool accepted = b.ok && b.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE;
        double weightedTime = std::numeric_limits<double>::infinity();
        if(accepted) {
          weightedTime = 1.0 / b.kernelsPerSecond;
        }
        const bool improvesBest = accepted && weightedTime < screenedBestTime;
        if(shouldPrintTunerResult(verboseTuner, improvesBest, accepted)) {
          out << "VulkanTuner:   attnCoopmat1Nhwc candidate tile=" << attentionCandidateString(c);
          if(b.ok && b.kernelsPerSecond > 0.0)
            out << " kps=" << b.kernelsPerSecond << " sec_per_kernel=" << 1.0 / b.kernelsPerSecond;
          if(std::isfinite(weightedTime))
            out << " weighted_us=" << (int)(weightedTime * 1.0e6);
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " result=" << (accepted ? (improvesBest ? "new_best" : "accepted") : "rejected") << endl;
        }
        if(accepted) {
          if(improvesBest)
            screenedBestTime = weightedTime;
          scored.push_back({c, weightedTime, rmse});
        }
      }
      std::sort(
        scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.weightedTime < b.weightedTime; });
      if(scored.size() > 3)
        scored.resize(3);

      double bestWeightedTime = std::numeric_limits<double>::infinity();
      AttentionCoopmatCandidate best{};
      double bestRmse = 0.0;
      for(const Scored& candidate: scored) {
        std::vector<double> wSamples;
        size_t rmseOk = 0;
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcCoopmat1WorkgroupSize = candidate.candidate.workgroupSize;
        cfg.attnNhwcCoopmat1BlockQ = candidate.candidate.blockQ;
        cfg.attnNhwcCoopmat1BlockKV = candidate.candidate.blockKV;
        cfg.attnNhwcCoopmat1TM = candidate.candidate.tm;
        cfg.attnNhwcCoopmat1TN = candidate.candidate.tn;
        cfg.attnNhwcCoopmat1TK = candidate.candidate.tk;
        cfg.attnNhwcCoopmat1SubgroupSize = candidate.candidate.subgroupSize;
        cfg.attnNhwcCoopmat1DirectKV = candidate.candidate.directKV;
        for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
          KernelBench b = coop->bench(ctx, cfg, candidate.candidate.kvChunkCount, iters);
          if(!b.ok || b.kernelsPerSecond <= 0.0)
            continue;
          double rmse = normalizedRmse(scalarRef.output, b.output);
          if(!(std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE))
            continue;
          double wt = 1.0 / b.kernelsPerSecond;
          wSamples.push_back(wt);
          rmseOk++;
        }
        double wt = medianValue(wSamples);
        if(verboseTuner || rmseOk != TUNER_CONFIRM_REPEATS)
          out << "VulkanTuner:   attnCoopmat1Nhwc confirm tile=" << attentionCandidateString(candidate.candidate)
              << " us=" << (wt > 0.0 ? (int)(wt * 1.0e6) : 0) << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS
              << endl;
        if(wt > 0.0 && wt < bestWeightedTime) {
          bestWeightedTime = wt;
          best = candidate.candidate;
          bestRmse = candidate.rmse;
        }
      }

      double bestCoopKps = bestWeightedTime < std::numeric_limits<double>::infinity() ? 1.0 / bestWeightedTime : 0.0;
      double bestSplitTime = std::numeric_limits<double>::infinity();
      AttentionCoopmatCandidate bestSplit{};
      double bestSplitRmse = 0.0;
      double bestSplitFlopsPerDispatch = 0.0;
      tunedConfig.attnNhwcCoopmat1TimeUs = bestCoopKps > 0.0 ? (int32_t)std::llround(bestWeightedTime * 1.0e6) : 0;
      if(bestCoopKps > 0.0) {
        tunedConfig.attnNhwcCoopmat1WorkgroupSize = best.workgroupSize;
        tunedConfig.attnNhwcCoopmat1BlockQ = best.blockQ;
        tunedConfig.attnNhwcCoopmat1BlockKV = best.blockKV;
        tunedConfig.attnNhwcCoopmat1TM = best.tm;
        tunedConfig.attnNhwcCoopmat1TN = best.tn;
        tunedConfig.attnNhwcCoopmat1TK = best.tk;
        tunedConfig.attnNhwcCoopmat1SubgroupSize = best.subgroupSize;
        tunedConfig.attnNhwcCoopmat1DirectKV = best.directKV;
        // A valid split-K entry may deliberately retain chunkCount=1 when no
        // split candidate is viable or faster for this device.
        tunedConfig.markKernelTuned({TUNED_ATTN_COOPMAT1_SPLITK});

        // Split-K is a separate batch-1 role. Sweep both its tile and chunk
        // count independently so it cannot compromise the regular path.
        VulkanKernels::AttentionTiled scalarB1(
          1, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);
        std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> coopB1 = makeCoopBench(1);
        bestSplitFlopsPerDispatch = coopB1->estimatedFlopsPerDispatch();
        KernelBench scalarRefB1 = scalarB1.bench(ctx, tunedConfig, iters);
        struct SplitScored {
          AttentionCoopmatCandidate candidate;
          double time;
          double rmse;
        };
        std::vector<SplitScored> splitScored;
        double screenedBestSplitTime = std::numeric_limits<double>::infinity();
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
              splitCfg.attnNhwcCoopmat1WorkgroupSize = split.workgroupSize;
              splitCfg.attnNhwcCoopmat1BlockQ = split.blockQ;
              splitCfg.attnNhwcCoopmat1BlockKV = split.blockKV;
              splitCfg.attnNhwcCoopmat1TM = split.tm;
              splitCfg.attnNhwcCoopmat1TN = split.tn;
              splitCfg.attnNhwcCoopmat1TK = split.tk;
              splitCfg.attnNhwcCoopmat1SubgroupSize = split.subgroupSize;
              splitCfg.attnNhwcCoopmat1DirectKV = split.directKV;
              if(!coopB1->validate(splitCfg, ctx.dev->info.properties.limits))
                continue;
              KernelBench result = coopB1->bench(ctx, splitCfg, split.kvChunkCount, iters);
              const double rmse =
                result.ok ? normalizedRmse(scalarRefB1.output, result.output) : std::numeric_limits<double>::infinity();
              const bool accepted =
                result.ok && result.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE;
              const double time = accepted ? 1.0 / result.kernelsPerSecond : std::numeric_limits<double>::infinity();
              const bool improvesBest = accepted && time < screenedBestSplitTime;
              if(shouldPrintTunerResult(verboseTuner, improvesBest, accepted)) {
                out << "VulkanTuner:   attnCoopmat1Nhwc split-k candidate tile=" << attentionCandidateString(split);
                if(result.ok && result.kernelsPerSecond > 0.0)
                  out << " kps=" << result.kernelsPerSecond << " sec_per_kernel=" << 1.0 / result.kernelsPerSecond;
                out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
                    << " result=" << (accepted ? (improvesBest ? "new_best" : "accepted") : "rejected") << endl;
              }
              if(accepted) {
                if(improvesBest)
                  screenedBestSplitTime = time;
                splitScored.push_back({split, time, rmse});
              }
            }
          }
        }
        std::sort(splitScored.begin(), splitScored.end(), [](const SplitScored& a, const SplitScored& b) {
          return a.time < b.time;
        });
        if(splitScored.size() > 3)
          splitScored.resize(3);
        for(const SplitScored& candidate: splitScored) {
          std::vector<double> samples;
          size_t rmseOk = 0;
          VulkanTuneParams splitCfg = tunedConfig;
          splitCfg.attnNhwcCoopmat1WorkgroupSize = candidate.candidate.workgroupSize;
          splitCfg.attnNhwcCoopmat1BlockQ = candidate.candidate.blockQ;
          splitCfg.attnNhwcCoopmat1BlockKV = candidate.candidate.blockKV;
          splitCfg.attnNhwcCoopmat1TM = candidate.candidate.tm;
          splitCfg.attnNhwcCoopmat1TN = candidate.candidate.tn;
          splitCfg.attnNhwcCoopmat1TK = candidate.candidate.tk;
          splitCfg.attnNhwcCoopmat1SubgroupSize = candidate.candidate.subgroupSize;
          splitCfg.attnNhwcCoopmat1DirectKV = candidate.candidate.directKV;
          for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
            KernelBench result = coopB1->bench(ctx, splitCfg, candidate.candidate.kvChunkCount, iters);
            const double rmse =
              result.ok ? normalizedRmse(scalarRefB1.output, result.output) : std::numeric_limits<double>::infinity();
            if(result.ok && result.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE) {
              samples.push_back(1.0 / result.kernelsPerSecond);
              rmseOk++;
            }
          }
          const double time = medianValue(samples);
          if(verboseTuner || rmseOk != TUNER_CONFIRM_REPEATS)
            out << "VulkanTuner:   attnCoopmat1Nhwc split-k confirm tile="
                << attentionCandidateString(candidate.candidate) << " us=" << (time > 0.0 ? (int)(time * 1.0e6) : 0)
                << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS << endl;
          if(time > 0.0 && time < bestSplitTime) {
            bestSplitTime = time;
            bestSplit = candidate.candidate;
            bestSplitRmse = candidate.rmse;
          }
        }
        if(bestSplitTime < std::numeric_limits<double>::infinity()) {
          tunedConfig.markKernelTuned({TUNED_ATTN_COOPMAT1_SPLITK});
          tunedConfig.attnNhwcCoopmat1SplitKWorkgroupSize = bestSplit.workgroupSize;
          tunedConfig.attnNhwcCoopmat1SplitKBlockQ = bestSplit.blockQ;
          tunedConfig.attnNhwcCoopmat1SplitKBlockKV = bestSplit.blockKV;
          tunedConfig.attnNhwcCoopmat1SplitKTM = bestSplit.tm;
          tunedConfig.attnNhwcCoopmat1SplitKTN = bestSplit.tn;
          tunedConfig.attnNhwcCoopmat1SplitKTK = bestSplit.tk;
          tunedConfig.attnNhwcCoopmat1SplitKSubgroupSize = bestSplit.subgroupSize;
          tunedConfig.attnNhwcCoopmat1SplitKDirectKV = bestSplit.directKV;
          tunedConfig.attnNhwcCoopmat1SplitKKVChunkCount = bestSplit.kvChunkCount;

          // The two roles are tuned independently above: regular attention at
          // the requested batch size, and split-K at batch one. Now measure
          // those fixed winners at every intermediate batch to choose the
          // largest contiguous prefix where split-K wins. A cutoff rather than
          // a per-batch table keeps the saved configuration compact and makes
          // reuse at smaller batch sizes safe.
          VulkanTuneParams regularCfg = tunedConfig;
          VulkanTuneParams splitCfg = tunedConfig;
          splitCfg.attnNhwcCoopmat1WorkgroupSize = bestSplit.workgroupSize;
          splitCfg.attnNhwcCoopmat1BlockQ = bestSplit.blockQ;
          splitCfg.attnNhwcCoopmat1BlockKV = bestSplit.blockKV;
          splitCfg.attnNhwcCoopmat1TM = bestSplit.tm;
          splitCfg.attnNhwcCoopmat1TN = bestSplit.tn;
          splitCfg.attnNhwcCoopmat1TK = bestSplit.tk;
          splitCfg.attnNhwcCoopmat1SubgroupSize = bestSplit.subgroupSize;
          splitCfg.attnNhwcCoopmat1DirectKV = bestSplit.directKV;
          bool splitKWonEverySmallerBatch = true;
          for(int crossoverBatch = 1; crossoverBatch <= batchSize; crossoverBatch++) {
            std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> regularBench = makeCoopBench(crossoverBatch);
            std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> splitBench = makeCoopBench(crossoverBatch);
            std::vector<double> regularSamples;
            std::vector<double> splitSamples;
            for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
              KernelBench regularResult = regularBench->bench(ctx, regularCfg, 1, iters);
              if(regularResult.ok && regularResult.kernelsPerSecond > 0.0)
                regularSamples.push_back(1.0 / regularResult.kernelsPerSecond);
              KernelBench splitResult = splitBench->bench(ctx, splitCfg, bestSplit.kvChunkCount, iters);
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
                  << " selected=" << (splitKWonEverySmallerBatch && splitKWins ? "split-k" : "regular") << endl;
            if(splitKWonEverySmallerBatch && splitKWins)
              tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch = crossoverBatch;
            else
              splitKWonEverySmallerBatch = false;
          }
          out << "VulkanTuner:   attnCoopmat1Nhwc split-k cutoff batch="
              << tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch << endl;
        }
        out << "VulkanTuner:   attnCoopmat1Nhwc regular winner tile=" << attentionCandidateString(best)
            << " us=" << tunedConfig.attnNhwcCoopmat1TimeUs
            << " tflops=" << coop->estimatedFlopsPerDispatch() / bestWeightedTime / 1.0e12
            << " max_rmse=" << bestRmse << endl;
        if(bestSplitTime < std::numeric_limits<double>::infinity())
          out << "VulkanTuner:   attnCoopmat1Nhwc split-k winner tile=" << attentionCandidateString(bestSplit)
              << " us=" << (int32_t)std::llround(bestSplitTime * 1.0e6)
              << " tflops=" << bestSplitFlopsPerDispatch / bestSplitTime / 1.0e12
              << " max_rmse=" << bestSplitRmse << endl;
      }
      return tunedConfig.attnNhwcCoopmat1TimeUs > 0;
    }

    // Reuse the candidate sweep while running the maintenance1 SPIR-V. The
    // temporary base fields are never persisted: copy its independently timed
    // winners into the maintenance1 namespace and restore portable coopmat1.
    bool tuneAttentionCoopmatMaintenance1Nhwc(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& cfg) {
      if(ctx.dev == nullptr || !ctx.dev->info.supportsCoopmatMaintenance1)
        return false;
      const VulkanTuneParams saved = cfg;
      tuneAttentionCoopmat1Nhwc(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, cfg, true);
      const VulkanTuneParams winner = cfg;
      cfg = saved;
      cfg.markKernelTuned({TUNED_ATTN_MAINTENANCE1});
      cfg.attnNhwcCoopmatMaintenance1TimeUs = winner.attnNhwcCoopmat1TimeUs;
      cfg.attnNhwcCoopmatMaintenance1WorkgroupSize = winner.attnNhwcCoopmat1WorkgroupSize;
      cfg.attnNhwcCoopmatMaintenance1BlockQ = winner.attnNhwcCoopmat1BlockQ;
      cfg.attnNhwcCoopmatMaintenance1BlockKV = winner.attnNhwcCoopmat1BlockKV;
      cfg.attnNhwcCoopmatMaintenance1TM = winner.attnNhwcCoopmat1TM;
      cfg.attnNhwcCoopmatMaintenance1TN = winner.attnNhwcCoopmat1TN;
      cfg.attnNhwcCoopmatMaintenance1TK = winner.attnNhwcCoopmat1TK;
      cfg.attnNhwcCoopmatMaintenance1SubgroupSize = winner.attnNhwcCoopmat1SubgroupSize;
      cfg.attnNhwcCoopmatMaintenance1DirectKV = winner.attnNhwcCoopmat1DirectKV;
      if(winner.hasKernelTuned(TUNED_ATTN_COOPMAT1_SPLITK))
        cfg.markKernelTuned({TUNED_ATTN_MAINTENANCE1_SPLITK});
      else
        cfg.clearKernelTuned({TUNED_ATTN_MAINTENANCE1_SPLITK});
      cfg.attnNhwcCoopmatMaintenance1SplitKWorkgroupSize = winner.attnNhwcCoopmat1SplitKWorkgroupSize;
      cfg.attnNhwcCoopmatMaintenance1SplitKBlockQ = winner.attnNhwcCoopmat1SplitKBlockQ;
      cfg.attnNhwcCoopmatMaintenance1SplitKBlockKV = winner.attnNhwcCoopmat1SplitKBlockKV;
      cfg.attnNhwcCoopmatMaintenance1SplitKTM = winner.attnNhwcCoopmat1SplitKTM;
      cfg.attnNhwcCoopmatMaintenance1SplitKTN = winner.attnNhwcCoopmat1SplitKTN;
      cfg.attnNhwcCoopmatMaintenance1SplitKTK = winner.attnNhwcCoopmat1SplitKTK;
      cfg.attnNhwcCoopmatMaintenance1SplitKSubgroupSize = winner.attnNhwcCoopmat1SplitKSubgroupSize;
      cfg.attnNhwcCoopmatMaintenance1SplitKDirectKV = winner.attnNhwcCoopmat1SplitKDirectKV;
      cfg.attnNhwcCoopmatMaintenance1SplitKKVChunkCount = winner.attnNhwcCoopmat1SplitKKVChunkCount;
      cfg.attnNhwcCoopmatMaintenance1SplitKCutoffBatch = winner.attnNhwcCoopmat1SplitKCutoffBatch;
      out << "VulkanTuner:   attnCoopmatMaintenance1Nhwc winner us=" << cfg.attnNhwcCoopmatMaintenance1TimeUs
          << " cutoff=" << cfg.attnNhwcCoopmatMaintenance1SplitKCutoffBatch << endl;
      return cfg.attnNhwcCoopmatMaintenance1TimeUs > 0;
    }

    bool tuneAttentionCoopmat2AccF32Nhwc(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.markKernelTuned({TUNED_ATTN_COOPMAT2});
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsCoopmat2Attention)
        return false;
      AttentionTuneProblem p = findAttentionTuneProblem(modelDesc);
      if(p.headDim <= 0)
        return false;
      const int validTokens = nnXLen * nnYLen;
      const int seqLen = roundUpToMultipleInt(validTokens, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);
      VulkanKernels::AttentionCoopmat2AccF32Bench coop(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);
      KernelBench ref = scalar.bench(ctx, tunedConfig, iters);
      if(!ref.ok || ref.kernelsPerSecond <= 0)
        return false;
      auto matches = [&](int bs, int bq, int bkv) {
        bool qk = false, pv = false;
        for(const Coopmat2FlexShape& s: ctx.dev->info.coopmat2FlexShapes) {
          if(
            (uint32_t)bs != s.workgroupInvocations || s.mGranularity == 0 || s.nGranularity == 0 || s.kGranularity == 0)
            continue;
          qk = qk || (bq % (int)s.mGranularity == 0 && bkv % (int)s.nGranularity == 0 &&
                      p.headDim % (int)s.kGranularity == 0);
          pv = pv || (bq % (int)s.mGranularity == 0 && p.vHeadDim % (int)s.nGranularity == 0 &&
                      bkv % (int)s.kGranularity == 0);
        }
        return qk && pv;
      };
      struct C {
        int bs, bq, bkv;
        double kps, rmse;
      };
      std::vector<C> tried, good;
      double screenedBestKps = 0.0;
      // The b7 and b11 sweeps both select the compact 4x/8x Q-block classes
      // with a 64-wide KV block. Keep 32 as the smaller fallback: 8/16 do far
      // less work per QK/PV loop iteration and are either slower or invalid on
      // the flexible shapes that can use the highest-throughput 64-wide tile.
      static const int bqMultipliers[] = {4, 8};
      const int maxBq = std::min(256, seqLen);
      for(const Coopmat2FlexShape& shape: ctx.dev->info.coopmat2FlexShapes) {
        const int bs = (int)shape.workgroupInvocations;
        std::vector<int> bqs = {bs};
        for(int mul: bqMultipliers)
          bqs.push_back((int)shape.mGranularity * mul);
        for(int bq: bqs)
          for(int bkv: {32, 64}) {
            if(bq <= 0 || bq > maxBq || !matches(bs, bq, bkv))
              continue;
            bool duplicate = false;
            for(const C& c: tried)
              if(c.bs == bs && c.bq == bq && c.bkv == bkv) {
                duplicate = true;
                break;
              }
            if(duplicate)
              continue;
            tried.push_back({bs, bq, bkv, 0.0, 0.0});
            VulkanTuneParams cfg = tunedConfig;
            cfg.attnNhwcCoopmat2WorkgroupSize = bs;
            cfg.attnNhwcCoopmat2BlockQ = bq;
            cfg.attnNhwcCoopmat2BlockKV = bkv;
            if(
              !coop.validate(cfg, ctx.dev->info.properties.limits) ||
              VulkanKernels::AttentionCoopmat2AccF32Nhwc::sharedBytes(bq, bkv, p.headDim, p.vHeadDim) +
                  ctx.dev->info.coopmat2ReservedSharedBytes >
                ctx.dev->info.properties.limits.maxComputeSharedMemorySize)
              continue;
            KernelBench b = coop.bench(ctx, cfg, iters);
            double rmse = b.ok ? normalizedRmse(ref.output, b.output) : std::numeric_limits<double>::infinity();
            const bool accept = b.ok && b.kernelsPerSecond > 0 && std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE;
            const bool improvesBest = accept && b.kernelsPerSecond > screenedBestKps;
            if(shouldPrintTunerResult(verboseTuner, improvesBest, accept))
              out << "VulkanTuner:   attnCoopmat2Nhwc candidate tile=[bs=" << bs << " bq=" << bq << " bkv=" << bkv
                  << "] kps=" << b.kernelsPerSecond << " rmse=" << rmse
                  << " result=" << (accept ? (improvesBest ? "new_best" : "accepted") : "rejected") << endl;
            if(accept) {
              if(improvesBest)
                screenedBestKps = b.kernelsPerSecond;
              good.push_back({bs, bq, bkv, b.kernelsPerSecond, rmse});
            }
          }
      }
      std::sort(good.begin(), good.end(), [](const C& a, const C& b) { return a.kps > b.kps; });
      tunedConfig.attnNhwcCoopmat2TimeUs = good.empty() ? 0 : (int32_t)std::llround(1e6 / good[0].kps);
      if(!good.empty()) {
        tunedConfig.attnNhwcCoopmat2WorkgroupSize = good[0].bs;
        tunedConfig.attnNhwcCoopmat2BlockQ = good[0].bq;
        tunedConfig.attnNhwcCoopmat2BlockKV = good[0].bkv;
        out << "VulkanTuner:   attnCoopmat2Nhwc winner tile=[bs=" << good[0].bs << " bq=" << good[0].bq
            << " bkv=" << good[0].bkv << "] us=" << tunedConfig.attnNhwcCoopmat2TimeUs
            << " tflops=" << coop.estimatedFlopsPerDispatch() * good[0].kps / 1.0e12
            << " max_rmse=" << good[0].rmse << endl;
      }
      return !good.empty();
    }

    struct AttentionDot2Candidate {
      int32_t workgroupSize, blockQ, blockKV;
    };

    string attentionDot2CandidateString(const AttentionDot2Candidate& c) {
      return Global::strprintf("[bs=%d bq=%d bkv=%d]", c.workgroupSize, c.blockQ, c.blockKV);
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
      tunedConfig.markKernelTuned({TUNED_ATTN_DOT2});
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsDot2F16)
        return false;
      AttentionTuneProblem p = findAttentionTuneProblem(modelDesc);
      if(p.headDim <= 0)
        return false;
      const int validTokens = nnXLen * nnYLen;
      const int seqLen = roundUpToMultipleInt(validTokens, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);
      VulkanKernels::AttentionDot2AccF32Bench dot2(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads, p.useRope, p.learnableRope, validTokens);

      KernelBench scalarRef = scalar.bench(ctx, tunedConfig, iters);
      if(!scalarRef.ok || scalarRef.kernelsPerSecond <= 0.0) {
        out << "VulkanTuner: attention DOT2 reference bench failed; keeping tiled attention." << endl;
        return false;
      }

      std::vector<AttentionDot2Candidate> candidates;
      for(int32_t workgroupSize: {32, 64, 128, 256}) {
        for(int32_t blockKV: {16, 32, 64}) {
          AttentionDot2Candidate c{workgroupSize, workgroupSize, blockKV};
          VulkanTuneParams cfg = tunedConfig;
          cfg.attnNhwcDot2WorkgroupSize = c.workgroupSize;
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
      out << "VulkanTuner: attention DOT2 AccF32 candidates=" << candidates.size() << endl;
      for(const AttentionDot2Candidate& c: candidates) {
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcDot2WorkgroupSize = c.workgroupSize;
        cfg.attnNhwcDot2BlockQ = c.blockQ;
        cfg.attnNhwcDot2BlockKV = c.blockKV;
        KernelBench b = dot2.bench(ctx, cfg, iters);
        double rmse = b.ok ? normalizedRmse(scalarRef.output, b.output) : std::numeric_limits<double>::infinity();
        bool accepted = b.ok && b.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE;
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

      double bestDot2Kps = 0.0;
      AttentionDot2Candidate best{};
      double bestRmse = 0.0;
      for(const Scored& candidate: scored) {
        std::vector<double> samples;
        size_t rmseOk = 0;
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcDot2WorkgroupSize = candidate.candidate.workgroupSize;
        cfg.attnNhwcDot2BlockQ = candidate.candidate.blockQ;
        cfg.attnNhwcDot2BlockKV = candidate.candidate.blockKV;
        for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
          KernelBench b = dot2.bench(ctx, cfg, iters);
          if(!b.ok || b.kernelsPerSecond <= 0.0)
            continue;
          double rmse = normalizedRmse(scalarRef.output, b.output);
          if(std::isfinite(rmse) && rmse <= MAX_RMSE_TOLERANCE) {
            samples.push_back(b.kernelsPerSecond);
            rmseOk++;
          }
        }
        double kps = medianValue(samples);
        out << "VulkanTuner:   attnDot2AccF32Nhwc confirm tile=" << attentionDot2CandidateString(candidate.candidate)
            << " kps=" << kps << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS << endl;
        if(kps > bestDot2Kps) {
          bestDot2Kps = kps;
          best = candidate.candidate;
          bestRmse = candidate.rmse;
        }
      }

      tunedConfig.attnNhwcDot2TimeUs = bestDot2Kps > 0.0 ? (int32_t)std::llround(1.0e6 / bestDot2Kps) : 0;
      if(bestDot2Kps > 0.0) {
        tunedConfig.attnNhwcDot2WorkgroupSize = best.workgroupSize;
        tunedConfig.attnNhwcDot2BlockQ = best.blockQ;
        tunedConfig.attnNhwcDot2BlockKV = best.blockKV;
        out << "VulkanTuner:   attnDot2AccF32Nhwc winner tile=" << attentionDot2CandidateString(best)
            << " us=" << tunedConfig.attnNhwcDot2TimeUs
            << " tflops=" << dot2.estimatedFlopsPerDispatch() * bestDot2Kps / 1.0e12
            << " max_rmse=" << bestRmse << endl;
      }
      return tunedConfig.attnNhwcDot2TimeUs > 0;
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
    const bool fp16Ready = ctx->fp16Storage && ctx->fp16Compute;

    const int64_t required = requiredKernelMask(modelDesc, ctx->dev->info, ctx->fp16Storage, ctx->fp16Compute, full);
    int64_t missing = full ? required : (required & ~tunedConfig.tunedKernelMask);
    if(missing == 0)
      return;

    auto kernels = makeMicroKernels(modelDesc, nnXLen, nnYLen, batchSize, tunedConfig);
    // The Winograd transforms are re-tuned together with the GEMM variant they
    // pack for (see retuneNhwcWinogradTransformsForCurrentLayout), not here.
    // The tiled fallbacks are skipped when any accelerated GEMM family is
    // usable, since they would never be dispatched.
    const int64_t dm = ctx->dev->info.disabledAccelVariantMask;
    const bool acceleratedGemmAvailable =
      fp16Ready && ((ctx->dev->info.supportsCoopmat1F16 && (dm & coopmat1AccF32Bits()) == 0) ||
                    (ctx->dev->info.supportsCoopmat1F16AccF16 && (dm & coopmat1AccF16Bits()) == 0) ||
                    (ctx->dev->info.supportsCoopmat2F16 && (dm & coopmat2AccF32Bits()) == 0) ||
                    (ctx->dev->info.supportsCoopmat2F16AccF16 && (dm & coopmat2AccF16Bits()) == 0) ||
                    (ctx->dev->info.supportsDot2F16 && (dm & dot2AccF32Bits()) == 0) ||
                    (ctx->dev->info.supportsDot2F16AccF16 && (dm & dot2AccF16Bits()) == 0));

    // Kept from the loop below so the tiled-attention fallback re-tune further
    // down doesn't have to rescan the kernel list.
    TunableKernel* attnTiledKernel = nullptr;

    for(auto& kernel: kernels) {
      const int64_t bit = kernel->tunedBit();
      if(bit == TUNED_WINOGRAD_TRANSFORM || bit == TUNED_WINOGRAD_UNTRANSFORM)
        continue;
      if(bit == TUNED_ATTN_TILED)
        attnTiledKernel = kernel.get();
      if(acceleratedGemmAvailable && (bit == TUNED_WINOGRAD_TILED || bit == TUNED_GEMM_STRIDED_TILED))
        continue;
      if((missing & bit) == 0)
        continue;

      const string name = kernel->name();
      out << endl << "VulkanTuner: tuning " << name << "..." << endl;
      if(bit == TUNED_WINOGRAD_TILED) {
        VulkanTuneParams tunedCandidate = tunedConfig;
        bool tuned = tuneWinogradGemmTiledMultiShape<VulkanKernels::WinogradGemm>(
          name, modelDesc, batchSize, nnXLen, nnYLen, tunedConfig, *ctx, iters, out, verboseTuner, tunedCandidate);
        if(tuned) {
          tunedConfig = tunedCandidate;
          continue;
        }
        out << "VulkanTuner: falling back to single-shape winogradGemmTiled tuning." << endl;
      }
      if(bit == TUNED_GEMM_STRIDED_TILED) {
        VulkanTuneParams tunedCandidate = tunedConfig;
        bool tuned = tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiledNhwc>(
          name, modelDesc, batchSize, nnXLen, nnYLen, tunedConfig, *ctx, iters, out, verboseTuner, tunedCandidate);
        if(tuned) {
          tunedConfig = tunedCandidate;
          continue;
        }
        out << "VulkanTuner: falling back to single-shape gemmStridedTiledNhwc tuning." << endl;
      }
      // Like the accelerated attention tiers, the tiled fallback records its
      // validity as soon as it is evaluated; its params are always
      // config-supported (defaults included).
      if(bit == TUNED_ATTN_TILED)
        tunedConfig.markKernelTuned({TUNED_ATTN_TILED});
      std::vector<TunedCandidate> top = tuneOne(*kernel, tunedConfig, *ctx, iters, out, verboseTuner);
      if(!top.empty())
        applyCandidate(top[0], tunedConfig);
    }

    if((missing & attentionVariantBits()) != 0) {
      // This is a composite tier: enabling an additional accelerator family
      // must benchmark only that missing family, then compare its saved time
      // against the already-tuned families. Do not discard a valid coopmat1
      // measurement merely because coopmat2 became eligible.
      const bool attnCoop1 =
        ctx->dev->info.supportsCoopmat1F16 && (dm & (TUNED_ATTN_COOPMAT1 | TUNED_ATTN_COOPMAT1_SPLITK)) == 0;
      const bool attnMaint1 = ctx->dev->info.supportsCoopmatMaintenance1 &&
                              (dm & (TUNED_ATTN_MAINTENANCE1 | TUNED_ATTN_MAINTENANCE1_SPLITK)) == 0;
      const bool attnCoop2 = ctx->dev->info.supportsCoopmat2Attention && (dm & TUNED_ATTN_COOPMAT2) == 0;
      const bool attnDot2 = ctx->dev->info.supportsDot2F16 && (dm & TUNED_ATTN_DOT2) == 0;
      const bool coopmat1Missing =
        attnCoop1 && (full || !tunedConfig.allOfKernelTuned({TUNED_ATTN_COOPMAT1, TUNED_ATTN_COOPMAT1_SPLITK}));
      const bool coopmat2Missing = attnCoop2 && (full || !tunedConfig.hasKernelTuned(TUNED_ATTN_COOPMAT2));
      if(coopmat1Missing) {
        out << endl << "VulkanTuner: tuning attention NHWC coopmat1 tier..." << endl;
        tuneAttentionCoopmat1Nhwc(*ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      }
      const bool maintenance1Missing =
        attnMaint1 &&
        (full || !tunedConfig.allOfKernelTuned({TUNED_ATTN_MAINTENANCE1, TUNED_ATTN_MAINTENANCE1_SPLITK}));
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
      // Coopmat > dot2 > tiled: a supported coopmat tier that produces no
      // valid candidate must fall back to dot2 (and then tiled) rather than
      // skipping straight to tiled. "Success" is a measured timeUs, not the
      // validity bit (which is recorded as soon as a tier is attempted).
      const bool one = attnCoop1 && tunedConfig.attnNhwcCoopmat1TimeUs > 0;
      const bool two = attnCoop2 && tunedConfig.attnNhwcCoopmat2TimeUs > 0;
      const bool maint = attnMaint1 && tunedConfig.attnNhwcCoopmatMaintenance1TimeUs > 0;
      // Selection is derived at runtime from these measured times; nothing to
      // persist here.
      if(full || !(one || two || maint)) {
        if(attnDot2) {
          out << endl << "VulkanTuner: tuning attention NHWC DOT2 AccF32 tier..." << endl;
          tuneAttentionDot2AccF32Nhwc(
            *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
      }
    }

    // When an accelerated attention family is present, the tiled fallback is
    // skipped from the micro-kernel loop above (it is not in the required mask)
    // because it is normally never dispatched. But every accelerated tier can
    // still fail for this model/device -- shape/shared-memory/fragment-budget
    // rejection or a failed bench -- in which case the runtime falls back to
    // the tiled kernel. Tune it then so the fallback has tuned (not merely
    // default) block sizes. Success is a measured timeUs, not the validity bit.
    if(!tunedConfig.hasKernelTuned(TUNED_ATTN_TILED)) {
      const bool anyAcceleratedTuned =
        (ctx->dev->info.supportsCoopmat1F16 && (dm & (TUNED_ATTN_COOPMAT1 | TUNED_ATTN_COOPMAT1_SPLITK)) == 0 &&
         tunedConfig.attnNhwcCoopmat1TimeUs > 0) ||
        (ctx->dev->info.supportsCoopmatMaintenance1 &&
         (dm & (TUNED_ATTN_MAINTENANCE1 | TUNED_ATTN_MAINTENANCE1_SPLITK)) == 0 &&
         tunedConfig.attnNhwcCoopmatMaintenance1TimeUs > 0) ||
        (ctx->dev->info.supportsCoopmat2Attention && (dm & TUNED_ATTN_COOPMAT2) == 0 &&
         tunedConfig.attnNhwcCoopmat2TimeUs > 0) ||
        (ctx->dev->info.supportsDot2F16 && (dm & TUNED_ATTN_DOT2) == 0 && tunedConfig.attnNhwcDot2TimeUs > 0);
      if(!anyAcceleratedTuned && attnTiledKernel != nullptr) {
        out << endl
            << "VulkanTuner: no accelerated attention tier measured; tuning " << attnTiledKernel->name() << "..."
            << endl;
        tunedConfig.markKernelTuned({TUNED_ATTN_TILED});
        std::vector<TunedCandidate> top = tuneOne(*attnTiledKernel, tunedConfig, *ctx, iters, out, verboseTuner);
        if(!top.empty())
          applyCandidate(top[0], tunedConfig);
      }
    }

    const int64_t gemmFamilies =
      stridedGemmFlavorBits() | winogradGemmFlavorBits() | conv3x3FlavorBits() | conv5x5FlavorBits();
    // GEMM-variant selection and the Winograd transform re-tune are one
    // operation: the transforms depend on the selected variant, so
    // tuneGemmVariantsForCurrentHardware re-tunes them for the finalized
    // layout right after selection. The gate fires whenever a GEMM family or
    // the transform bits themselves are missing (reconcile invalidates them on
    // a stale layout), so reconcile-cleared transforms get re-tuned here.
    if((missing & (gemmFamilies | TUNED_WINOGRAD_TRANSFORM | TUNED_WINOGRAD_UNTRANSFORM)) != 0)
      tuneGemmVariantsForCurrentHardware(
        *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, full, tunedConfig);
    tunedConfig.tunedKernelMask |= required & ~(TUNED_WINOGRAD_TRANSFORM | TUNED_WINOGRAD_UNTRANSFORM);
    // The Winograd conv path is only marked usable when the winograd GEMM
    // variant that will actually be selected (coopmat/dot2/tiled) is valid.
    // This is derived from the measured variants rather than assumed.
    if((required & (TUNED_CONV3X3_WINOGRAD | TUNED_CONV5X5_WINOGRAD)) != 0) {
      const bool winogradGemmValid =
        tunedConfig.hasKernelTuned(TUNED_WINOGRAD_TILED) ||
        (ctx->dev != nullptr && fp16Ready &&
         resolveWinogradGemmVariant(ctx->dev->info, tunedConfig, true) != TUNED_WINOGRAD_TILED);
      if(!winogradGemmValid)
        tunedConfig.clearKernelTuned(
          {TUNED_CONV3X3_WINOGRAD, TUNED_CONV5X5_WINOGRAD, TUNED_WINOGRAD_TRANSFORM, TUNED_WINOGRAD_UNTRANSFORM});
    }

    // Recompute the "tuned but no valid score" bits from the measured times, so
    // an incremental run skips failed tiers (they fall back to the next tier,
    // which was tuned above) and a full run retries them.
    tunedConfig.invalidKernelMask =
      (tunedConfig.invalidKernelMask & ~allAccelVariantBits()) | computeInvalidVariantBitsFromTimeUs(tunedConfig);
  }

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND

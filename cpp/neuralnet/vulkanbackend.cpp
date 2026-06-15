#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkanbackend.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <vector>
#include "../core/fileutils.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/makedir.h"
#include "../core/simpleallocator.h"
#include "../core/test.h"
#include "../core/using.h"
#include "../dataio/homedata.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkankernels.h"
#include "../neuralnet/vulkanlayers.h"
#include "../neuralnet/vulkantuner.h"

using namespace VulkanHelpers;
using half_t = half_float::half;
using std::lock_guard;
using std::mutex;

// ============================================================================
// FP16 CONVENTIONS (same as OpenCL backend)
//
// IMPORTANT: "useFP16" / "usingFP16Storage" / the USE_FP16_STORAGE shader spec
// constant all mean FP16 *storage*. Tensors are stored as 16-bit floats and most
// shaders widen loads to FP32, do arithmetic in FP32, and narrow stores back to
// FP16. FP16 compute is the exception, not the rule.
//
// A few accelerator kernels instead keep the math in FP16 with an FP16 *internal
// accumulator*: the coopmat1/coopmat2 (VK_KHR / VK_NV cooperative_matrix) and
// dot2 (VK_NV_dot_product) GEMM/conv/Winograd variants whose names end in
// "AccF16". These still read and write FP16 storage, but accumulate within the
// cooperative matrix in FP16 rather than widening to FP32.
//
// FP16 accumulation trades a little precision for speed, so the tuner only picks
// it when it pays off. It benchmarks each FP16-accumulator variant head-to-head
// against its FP32-accumulator equivalent and selects FP16 only when it is at
// least 25% faster (its measured time is ranked with a
// VARIANT_SELECT_PENALTY_ACCF16 = 1.25x multiplier, so it only wins if
// time_us_f16 * 1.25 < time_us_f32); otherwise the FP32
// accumulator wins. Variant availability is additionally gated on device support
// (shaderFloat16 plus the relevant cooperative-matrix / dot-product features).
//
// The KATAGO_VULKAN_ACCEL_VARIANTS environment variable can selectively enable
// or disable the FP32 and FP16 variants (e.g. "dot2,accf16" or
// "-coopmat2accf16"), which the tuner and the runtime selection both honor. The
// FP16-accumulation dot2 variant is opt-in by default and must be explicitly
// enabled this way.
//
// When using FP16:
//   - Every spatial trunk tensor and mask is in FP16.
//   - Batch norm scales and biases are in FP16.
//   - Everything else stays FP32:
//     -- Initial MatMul for global features
//     -- Global pooling outputs
//     -- Value head and policy head global pooling
//     -- All MatMul and MatBias layers (fully-connected) are FP32
// ============================================================================

// Vulkan accepts either public input layout. Native convolutional graphs keep
// NHWC throughout; external channel-first (NCHW) inputs convert once on-device
// at the model boundary.

// ============================================================================
// Section: Global Instance and Model File Lifecycle
// ============================================================================

static VkInstance g_vulkanInstance = VK_NULL_HANDLE;
static std::mutex g_instanceMutex;

// LoadedModel

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// Global init / cleanup

void NeuralNet::globalInitialize() {
  static_assert(sizeof(int) >= 4, "");
}

void NeuralNet::globalCleanup() {
  lock_guard<mutex> lock(g_instanceMutex);
  if(g_vulkanInstance != VK_NULL_HANDLE) {
    vkDestroyInstance(g_vulkanInstance, nullptr);
    g_vulkanInstance = VK_NULL_HANDLE;
  }
}

namespace {

  VkInstance getOrCreateInstance() {
    lock_guard<mutex> lock(g_instanceMutex);
    if(g_vulkanInstance != VK_NULL_HANDLE)
      return g_vulkanInstance;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "KataGo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

#ifdef DEBUG
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = layers;
#endif

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &g_vulkanInstance));
    return g_vulkanInstance;
  }

  // Hex-encode a byte blob (used for pipelineCacheUUID → cache/tune filename).
  string bytesToHex(const uint8_t* data, size_t len) {
    static constexpr char digits[] = "0123456789abcdef";
    string s;
    s.resize(len * 2);
    for(size_t i = 0; i < len; i++) {
      s[i * 2 + 0] = digits[(data[i] >> 4) & 0xF];
      s[i * 2 + 1] = digits[data[i] & 0xF];
    }
    return s;
  }

  // Stable per-driver-per-physical-device identity string. pipelineCacheUUID is
  // generated by the ICD from vendor + device + driver version + any bits that
  // invalidate cached pipelines; two physical GPUs of the same model report
  // different UUIDs while sharing the same marketing deviceName.
  string deviceUuidHex(const InitializedVulkanDevice* dev) {
    return bytesToHex(dev->info.properties.pipelineCacheUUID, VK_UUID_SIZE);
  }

  // Tuning-identity key: (vendorID, deviceID, driverVersion). Same across two
  // physically distinct GPUs of the same model on the same host — which is what
  // we want, because tuning results are architecture-level and reusing them is
  // the whole point of caching. Distinct from pipelineCacheUUID, which varies
  // per physical device and would over-partition the tuning cache.
  string tuningDeviceKey(const InitializedVulkanDevice* dev) {
    const auto& p = dev->info.properties;
    return Global::strprintf("v%08x_d%08x_drv%08x", p.vendorID, p.deviceID, p.driverVersion);
  }

  bool vulkanBatchProfileEnabled() {
    static const bool cached = []() {
      std::string_view v = getenvStringView("KATAGO_VULKAN_PROFILE_BATCHES");
      return !v.empty() && v[0] != '0';
    }();
    return cached;
  }

}  // namespace

void NeuralNet::printDevices() {
  VkInstance inst = getOrCreateInstance();
  VulkanHelpers::printDevices(inst, nullptr);
}

struct CompiledPipelines {
  VkDevice device;
  bool usingFP16Storage;
  bool usingFP16Compute;
  VulkanTuneParams tuneParams;
  VkPipelineCache pipelineCache;
  string pipelineCacheFile;
  Logger* logger;

  // CompiledPipelines holds the per-(device, ComputeContext) shared resources:
  // the driver-level VkPipelineCache (disk-backed) and frozen tuneParams. All
  // compute kernels are owned by individual layer/block/model structs and built
  // at load time via the per-kernel VulkanKernels::X::build() factories.

  void createPipelineCache() {
    vector<uint8_t> initialData;
    if(pipelineCacheFile != "") {
      std::ifstream in;
      if(FileUtils::tryOpen(in, pipelineCacheFile, std::ios::binary | std::ios::ate)) {
        std::streamsize size = in.tellg();
        if(size > 0) {
          in.seekg(0, std::ios::beg);
          initialData.resize(static_cast<size_t>(size));
          in.read(reinterpret_cast<char*>(initialData.data()), size);
          if(!in.good() && !in.eof())
            initialData.clear();
        }
        in.close();
      }
    }

    VkPipelineCacheCreateInfo cacheCI = {};
    cacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCI.initialDataSize = initialData.size();
    cacheCI.pInitialData = initialData.empty() ? nullptr : initialData.data();

    VkResult result = vkCreatePipelineCache(device, &cacheCI, nullptr, &pipelineCache);
    if(result != VK_SUCCESS && !initialData.empty()) {
      if(logger != nullptr)
        logger->write("Vulkan: failed to load pipeline cache, creating empty cache: " + pipelineCacheFile);
      cacheCI.initialDataSize = 0;
      cacheCI.pInitialData = nullptr;
      VK_CHECK(vkCreatePipelineCache(device, &cacheCI, nullptr, &pipelineCache));
    } else {
      VK_CHECK(result);
      if(logger != nullptr && !initialData.empty()) {
        logger->write(
          "Vulkan: loaded pipeline cache (" + Global::uint64ToString((uint64_t)initialData.size()) +
          " bytes): " + pipelineCacheFile);
      }
    }
  }

  void savePipelineCache() const {
    if(pipelineCache == VK_NULL_HANDLE || pipelineCacheFile == "")
      return;

    size_t size = 0;
    if(vkGetPipelineCacheData(device, pipelineCache, &size, nullptr) != VK_SUCCESS || size == 0)
      return;
    vector<uint8_t> data(size);
    if(vkGetPipelineCacheData(device, pipelineCache, &size, data.data()) != VK_SUCCESS)
      return;

    try {
      const string tmpFile = pipelineCacheFile + ".tmp";
      std::ofstream out;
      FileUtils::open(out, tmpFile, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)size);
      out.close();
      if(!FileUtils::tryRename(tmpFile, pipelineCacheFile)) {
        FileUtils::open(out, pipelineCacheFile, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)size);
        out.close();
      }
      if(logger != nullptr) {
        logger->write(
          "Vulkan: wrote pipeline cache (" + Global::uint64ToString((uint64_t)size) + " bytes): " + pipelineCacheFile);
      }
    } catch(const StringError&) {
      if(logger != nullptr)
        logger->write("Vulkan: warning, unable to write pipeline cache to " + pipelineCacheFile);
    }
  }

  CompiledPipelines(
    VkDevice dev,
    bool fp16Storage,
    bool fp16Compute,
    const string& pipelineCacheFile_,
    Logger* logger_,
    const VulkanTuneParams& tuneParams_)
    : device(dev),
      usingFP16Storage(fp16Storage),
      usingFP16Compute(fp16Compute),
      tuneParams(tuneParams_),
      pipelineCache(VK_NULL_HANDLE),
      pipelineCacheFile(pipelineCacheFile_),
      logger(logger_) {
    createPipelineCache();
    // Only the shared pipeline cache is created here; all compute kernels are
    // owned per-layer/block and built at load time.
  }

  ~CompiledPipelines() {
    savePipelineCache();
    if(pipelineCache != VK_NULL_HANDLE) {
      vkDestroyPipelineCache(device, pipelineCache, nullptr);
      pipelineCache = VK_NULL_HANDLE;
    }
  }

  CompiledPipelines() = delete;
  CompiledPipelines(const CompiledPipelines&) = delete;
  CompiledPipelines& operator=(const CompiledPipelines&) = delete;
};

// ============================================================================
// Section: ComputeContext and Public Context Entry Points
// ============================================================================

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const std::string homeDataDirOverride;
  Logger* const logger;

  VkInstance instance;
  VulkanDevicesContext* devicesContext;
  std::map<VkPhysicalDevice, CompiledPipelines*> compiledPipelinesByDevice;

  // Set true the first time NeuralNet::createComputeHandle runs against this
  // context. tuneSelf() refuses to run once this is set: by then the per-layer
  // pipelines have been compiled with the current spec constants, and changing
  // tuneParams would create a silent mismatch.
  bool handlesAlreadyCreated = false;

  // gpuIdxs whose tuneParams came from compiled-in defaults because no cache
  // file resolved during ctor. tuneSelf consults this; needsTuning() reports it.
  std::vector<int> gpuIdxsNeedingTune;

  // Whether load-time auto-tuning was requested (a non-null modelDescForTuneLoad
  // was passed to the ctor), and whether to re-tune per board size.
  bool autoTuneEnabled = false;
  bool autoTuneReTunePerBoardSize = false;

  ComputeContext(
    VkInstance inst,
    const vector<int>& gpuIdxs,
    Logger* logger_,
    int nnX,
    int nnY,
    const string& homeDataDirOverride_,
    enabled_t fp16Mode,
    const ModelDesc* modelDescForTuneLoad = nullptr,
    const std::string& vulkanTunerFile = "",
    bool vulkanReTunePerBoardSize = false)
    : nnXLen(nnX),
      nnYLen(nnY),
      usingFP16Mode(fp16Mode),
      homeDataDirOverride(homeDataDirOverride_),
      logger(logger_),
      instance(inst),
      autoTuneEnabled(modelDescForTuneLoad != nullptr),
      autoTuneReTunePerBoardSize(vulkanReTunePerBoardSize) {
    auto allDeviceInfos = VulkanDeviceInfo::getAllDeviceInfosOnSystem(inst, logger);

    bool requestFP16Storage = (fp16Mode == enabled_t::True || fp16Mode == enabled_t::Auto);
    bool requestFP16Compute = requestFP16Storage;

    devicesContext = new VulkanDevicesContext(allDeviceInfos, gpuIdxs, requestFP16Storage, requestFP16Compute, logger);

    for(auto& dev: devicesContext->devicesToUse) {
      bool fp16Storage = requestFP16Storage && dev->info.supportsFP16Storage;
      bool fp16Compute = requestFP16Compute && dev->info.supportsFP16Compute;

      string pipelineCacheFile;
      {
        auto sanitize = [](const string& s) {
          string out;
          out.reserve(s.size());
          for(char c: s)
            if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
              out.push_back(c);
          return out;
        };
        string dir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkancache";
        MakeDir::make(dir);
        string gpuName = sanitize(dev->info.properties.deviceName);
        string uuid = deviceUuidHex(dev.get());
        pipelineCacheFile = Global::strprintf(
          "%s/pcache_%s_UUID-%s_x%dy%d_f16storage%d_f16compute%d.bin",
          dir.c_str(),
          gpuName.c_str(),
          uuid.c_str(),
          nnXLen,
          nnYLen,
          fp16Storage ? 1 : 0,
          fp16Compute ? 1 : 0);
      }

      // Resolve per-device tuneParams. Three modes:
      //   1. Explicit file (vulkanTunerFile non-empty): load it; if it is missing/invalid
      //      during normal model startup, ignore it and defer to a later tuneSelf() call.
      //   2. modelDesc given: try the per-(gpu,board,model) cache, then (if not per-board)
      //      the MAX_BOARD_LEN cache. On miss, defer to a later tuneSelf() call.
      //   3. modelDesc null: leave at compiled-in defaults silently (e.g. layer tests, or
      //      the standalone tuner, which loads its output file later once its path is known).
      VulkanTuneParams perDeviceParams;
      const string deviceName(dev->info.properties.deviceName);
      const string tuneKey = tuningDeviceKey(dev.get());

      // Queue this GPU for a lazy tune run. The tuner only re-tunes kernels
      // whose validity bit is missing (or the NHWC Winograd transforms, whose
      // layout signature reconcile may have cleared), so one queue entry covers
      // both a stale/incomplete file and a full tune.
      auto requestRetuneForDevice = [&]() {
        if(modelDescForTuneLoad == nullptr)
          return;
        if(
          std::find(gpuIdxsNeedingTune.begin(), gpuIdxsNeedingTune.end(), dev->info.gpuIdx) == gpuIdxsNeedingTune.end())
          gpuIdxsNeedingTune.push_back(dev->info.gpuIdx);
      };

      if(!vulkanTunerFile.empty()) {
        auto handleBadExplicitTuneFile = [&](const string& why) {
          if(modelDescForTuneLoad == nullptr)
            throw IOError(why);
          if(logger != nullptr)
            logger->write(
              "Vulkan tune file could not be used (" + why + "); ignoring and re-tuning: " + vulkanTunerFile);
          requestRetuneForDevice();
        };

        bool loadedExplicitFile = false;
        VulkanTuneParams loaded;
        try {
          loaded = VulkanTuneParams::load(vulkanTunerFile);
          loadedExplicitFile = true;
        } catch(const IOError& e) {
          handleBadExplicitTuneFile(e.what());
        }

        if(loadedExplicitFile) {
          // Single load-time validity gate: reset anything not supported by this
          // device/kernel to defaults and clear its tuned bit. Everything that
          // remains tuned is valid; whatever the model still needs but is missing
          // (or whose layout reconcile cleared) gets re-tuned lazily.
          const bool invalidated =
            VulkanTuner::reconcileTuneParamsForDevice(loaded, dev->info, logger, fp16Storage, fp16Compute);
          const int64_t requiredMask =
            VulkanTuner::requiredKernelMask(modelDescForTuneLoad, dev->info, fp16Storage, fp16Compute, false);
          if(invalidated || (loaded.tunedKernelMask & requiredMask) != requiredMask)
            requestRetuneForDevice();
          if(logger != nullptr) {
            logger->write("Loaded vulkan tuning parameters from: " + vulkanTunerFile);
            std::ostringstream s;
            VulkanTuner::appendTuneParamsSummary(s, loaded, modelDescForTuneLoad);
            logger->write(
              "Using vulkan tuning parameters for GPU " + Global::intToString(dev->info.gpuIdx) + " (" + deviceName +
              "): " + s.str());
          }
          perDeviceParams = loaded;
        }
      } else if(modelDescForTuneLoad != nullptr) {
        if(!tryLoadFromCache(
             dev->info, deviceName, tuneKey, modelDescForTuneLoad, fp16Storage, fp16Compute, perDeviceParams)) {
          gpuIdxsNeedingTune.push_back(dev->info.gpuIdx);
        } else {
          const int64_t requiredMask =
            VulkanTuner::requiredKernelMask(modelDescForTuneLoad, dev->info, fp16Storage, fp16Compute, false);
          if((perDeviceParams.tunedKernelMask & requiredMask) != requiredMask)
            gpuIdxsNeedingTune.push_back(dev->info.gpuIdx);
        }
      }

      CompiledPipelines* pipelines =
        new CompiledPipelines(dev->device, fp16Storage, fp16Compute, pipelineCacheFile, logger, perDeviceParams);
      compiledPipelinesByDevice[dev->info.physicalDevice] = pipelines;
    }
  }

  // Try the default-directory cache file for (device tuning-key, nnX, nnY, model).
  // On miss and !autoTuneReTunePerBoardSize, retry at MAX_BOARD_LEN. Returns true
  // if a valid file was loaded; out is left default-constructed otherwise.
  bool tryLoadFromCache(
    const VulkanDeviceInfo& deviceInfo,
    const string& deviceName,
    const string& tuneKey,
    const ModelDesc* modelDesc,
    bool fp16Storage,
    bool fp16Compute,
    VulkanTuneParams& out) const {
    auto attempt = [&](int xLen, int yLen) -> bool {
      string dir = VulkanTuner::defaultDirectory(homeDataDirOverride);
      string path = dir + "/" + VulkanTuner::defaultFileName(deviceName, tuneKey, xLen, yLen, modelDesc, fp16Storage);
      try {
        VulkanTuneParams loaded = VulkanTuneParams::load(path);
        // Single load-time validity gate: reset anything not supported by this
        // device/kernel to defaults and clear its tuned bit. A cache hit still
        // returns the (reconciled) params; the caller re-tunes whatever the
        // model needs that is now missing.
        VulkanTuner::reconcileTuneParamsForDevice(loaded, deviceInfo, logger, fp16Storage, fp16Compute);
        if(logger != nullptr) {
          logger->write("Loaded vulkan tuning parameters from: " + path);
          std::ostringstream s;
          VulkanTuner::appendTuneParamsSummary(s, loaded, modelDesc);
          logger->write("Using vulkan tuning parameters: " + s.str());
        }
        out = loaded;
        return true;
      } catch(const IOError&) {
        return false;
      }
    };
    if(attempt(nnXLen, nnYLen))
      return true;
    if(!autoTuneReTunePerBoardSize && (nnXLen != NNPos::MAX_BOARD_LEN || nnYLen != NNPos::MAX_BOARD_LEN)) {
      if(attempt(NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN))
        return true;
    }
    return false;
  }

  bool needsTuning() const { return !gpuIdxsNeedingTune.empty(); }

  void markHandleCreated() { handlesAlreadyCreated = true; }

  // Run the autotune sweep on each device whose tuneParams came from defaults,
  // mutate its CompiledPipelines::tuneParams in place, and persist to disk.
  // Forward-declared here; defined after VulkanWorkContext + TuningContext.
  void tuneSelf(
    const ModelDesc* modelDesc,
    int batchSize,
    int winograd3x3OutTile,
    int benchIters,
    bool verboseTuner,
    bool full,
    const std::string& saveFileOverride);

  ~ComputeContext() {
    for(auto& kv: compiledPipelinesByDevice)
      delete kv.second;
    delete devicesContext;
  }

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg) {
  if(gpuIdxs.empty())
    throw StringError("NeuralNet::createComputeContext - no GPUs specified");

  string vulkanTunerFile;
  if(cfg.contains("vulkanTunerFile"))
    vulkanTunerFile = cfg.getString("vulkanTunerFile");
  bool vulkanReTunePerBoardSize = false;
  if(cfg.contains("vulkanReTunePerBoardSize"))
    vulkanReTunePerBoardSize = cfg.getBool("vulkanReTunePerBoardSize");

  VkInstance inst = getOrCreateInstance();
  auto* ctx = new ComputeContext(
    inst,
    gpuIdxs,
    logger,
    nnXLen,
    nnYLen,
    homeDataDirOverride,
    useFP16Mode,
    &(loadedModel->modelDesc),
    vulkanTunerFile,
    vulkanReTunePerBoardSize);

  if(ctx->needsTuning()) {
    int tuneBatchSize = VulkanTuner::DEFAULT_BATCH_SIZE;
    if(cfg.contains("numSearchThreads")) {
      // Search batches typically range from 1 up to about numSearchThreads,
      // so tune at the upper end of the expected runtime range.
      tuneBatchSize = std::max(1, cfg.getInt("numSearchThreads", 1, 65536));
    }
    ctx->tuneSelf(
      &(loadedModel->modelDesc),
      tuneBatchSize,
      VulkanTuner::DEFAULT_WINOGRAD_3X3_TILE_SIZE,
      /*benchIters=*/500,
      /*verboseTuner=*/false,
      /*full=*/false,
      /*saveFileOverride=*/"");
  }

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* ctx) {
  delete ctx;
}

ComputeContext* VulkanTuner::createComputeContextForVulkanTuner(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode) {
  if(gpuIdxs.empty())
    throw StringError("VulkanTuner::createComputeContextForVulkanTuner - no GPUs specified");
  VkInstance inst = getOrCreateInstance();
  return new ComputeContext(
    inst,
    gpuIdxs,
    logger,
    nnXLen,
    nnYLen,
    homeDataDirOverride,
    useFP16Mode,
    /*modelDescForTuneLoad=*/nullptr,
    /*vulkanTunerFile=*/"",
    /*vulkanReTunePerBoardSize=*/false);
}

void NeuralNet::tuneVulkanComputeContext(
  ComputeContext* ctx,
  const ModelDesc* modelDesc,
  int batchSize,
  int winograd3x3OutTile,
  int benchIters,
  bool verboseTuner,
  bool full,
  const std::string& saveFileOverride) {
  ctx->tuneSelf(modelDesc, batchSize, winograd3x3OutTile, benchIters, verboseTuner, full, saveFileOverride);
}

// ============================================================================
// Section: VulkanWorkContext and TuningContext Utilities
// ============================================================================

// VulkanWorkContext: a device + one-shot command buffer + fence + optional
// timestamp query pool, owning a minimal ComputeContext. Provides command
// recording/submission, flat-buffer upload, and a GPU-or-wall-clock timing
// helper. Shared base for TestContext (isolated layer tests) and
// VulkanTuner::TuningContext (the autotuner). This is the one place that knows
// how to set up a device for ad-hoc dispatching and how to time it.

void VulkanWorkContext::init(int gpuIdx, int nnXLen, int nnYLen, enabled_t fp16Mode, Logger* logger) {
  VkInstance inst = getOrCreateInstance();
  computeCtx = new ComputeContext(inst, {gpuIdx}, logger, nnXLen, nnYLen, "", fp16Mode);
  ownsComputeCtx = true;
  initBorrowed(computeCtx, gpuIdx, logger);
}

void VulkanWorkContext::initBorrowed(ComputeContext* externalCtx, int gpuIdx, Logger* logger) {
  if(computeCtx == nullptr) {
    computeCtx = externalCtx;
    ownsComputeCtx = false;
  }
  // (If init() called us we already set computeCtx + ownsComputeCtx=true.)
  dev = computeCtx->devicesContext->findGpuExn(gpuIdx);
  pipelines = computeCtx->compiledPipelinesByDevice.at(dev->info.physicalDevice);

  VkCommandBufferAllocateInfo cbAI = {};
  cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAI.commandPool = dev->commandPool;
  cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAI.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(dev->device, &cbAI, &cmd));

  VkFenceCreateInfo fenceCI = {};
  fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(dev->device, &fenceCI, nullptr, &fence));

  // Timestamp capability: timestampValidBits is per queue family; timestampPeriod
  // is the always-present granularity. If bits == 0, the queue cannot write
  // timestamps, so never record one — fall back to wall-clock timing.
  const VkPhysicalDeviceProperties& props = dev->info.properties;
  timestampPeriodNs = props.limits.timestampPeriod;
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(dev->info.physicalDevice, &qfCount, nullptr);
  vector<VkQueueFamilyProperties> qfProps(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(dev->info.physicalDevice, &qfCount, qfProps.data());
  uint32_t validBits = 0;
  int cqf = dev->info.computeQueueFamilyIdx;
  if(cqf >= 0 && cqf < (int)qfCount)
    validBits = qfProps[cqf].timestampValidBits;
  useGpuTimestamps = (validBits > 0) && (timestampPeriodNs > 0.0);

  if(useGpuTimestamps) {
    VkQueryPoolCreateInfo qpCI = {};
    qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpCI.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(dev->device, &qpCI, nullptr, &queryPool));
  }

  if(logger != nullptr) {
    logger->write(
      "Vulkan: work context on " + string(props.deviceName) + " timestampPeriod=" +
      Global::doubleToString(timestampPeriodNs) + "ns timestampValidBits=" + Global::intToString((int)validBits) +
      (useGpuTimestamps ? " (GPU-timestamp timing)" : " (wall-clock timing fallback)"));
  }
}

void VulkanWorkContext::destroy() {
  if(dev == nullptr)
    return;
  if(queryPool != VK_NULL_HANDLE)
    vkDestroyQueryPool(dev->device, queryPool, nullptr);
  if(fence != VK_NULL_HANDLE)
    vkDestroyFence(dev->device, fence, nullptr);
  if(cmd != VK_NULL_HANDLE)
    vkFreeCommandBuffers(dev->device, dev->commandPool, 1, &cmd);
  queryPool = VK_NULL_HANDLE;
  fence = VK_NULL_HANDLE;
  cmd = VK_NULL_HANDLE;
  if(ownsComputeCtx)
    delete computeCtx;
  computeCtx = nullptr;
  ownsComputeCtx = false;
  dev = nullptr;
  pipelines = nullptr;
}

VkDevice VulkanWorkContext::device() const {
  return dev->device;
}

const VkPhysicalDeviceMemoryProperties& VulkanWorkContext::memProps() const {
  return dev->info.memoryProperties;
}

PFN_vkCmdPushDescriptorSetKHR VulkanWorkContext::getPushDescFn() const {
  return reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(dev->device, "vkCmdPushDescriptorSetKHR"));
}

void VulkanWorkContext::beginRecording() {
  VkCommandBufferBeginInfo bi = {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
}

void VulkanWorkContext::submitAndWait() {
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  {
    lock_guard<mutex> lk(const_cast<mutex&>(dev->queueMutex));
    VK_CHECK(vkQueueSubmit(dev->computeQueue, 1, &si, fence));
  }
  VK_CHECK(vkWaitForFences(dev->device, 1, &fence, VK_TRUE, UINT64_MAX));
  VK_CHECK(vkResetFences(dev->device, 1, &fence));
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
}

void VulkanWorkContext::submitCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
  beginRecording();
  VkBufferCopy r = {0, 0, size};
  vkCmdCopyBuffer(cmd, src, dst, 1, &r);
  submitAndWait();
}

VBuf VulkanWorkContext::makeInputBuf(const vector<float>& data) {
  auto buf = makeDeviceBuf(device(), memProps(), std::max<size_t>(data.size(), 1), false);
  VkDeviceSize byteSize = data.size() * sizeof(float);
  if(byteSize == 0)
    return buf;
  auto staging = VulkanBuffer(allocateBuffer(
    device(),
    memProps(),
    byteSize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  void* mapped;
  VK_CHECK(vkMapMemory(device(), staging.memory, 0, byteSize, 0, &mapped));
  memcpy(mapped, data.data(), byteSize);
  vkUnmapMemory(device(), staging.memory);
  submitCopy(staging.buffer, buf->buffer, byteSize);
  return buf;
}

void VulkanWorkContext::downloadFloats(VulkanBuffer* srcBuf, vector<float>& dst, size_t numElts) {
  VkDeviceSize stagingSize = numElts * sizeof(float);
  if(stagingSize == 0)
    return;
  auto staging = VulkanBuffer(allocateBuffer(
    device(),
    memProps(),
    stagingSize,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  beginRecording();
  VkBufferCopy r = {0, 0, stagingSize};
  vkCmdCopyBuffer(cmd, srcBuf->buffer, staging.buffer, 1, &r);
  submitAndWait();
  void* mapped;
  VK_CHECK(vkMapMemory(device(), staging.memory, 0, stagingSize, 0, &mapped));
  memcpy(dst.data(), mapped, stagingSize);
  vkUnmapMemory(device(), staging.memory);
}

double VulkanWorkContext::timeDispatches(int iters, const std::function<void()>& recordOne) {
  // Split into small submissions to avoid GPU watchdog timeouts (TDR).
  // Each chunk is one command buffer; wall-clock time is accumulated across chunks.
  // The timestamp path also chunks — one query pool pair per submission.
  static constexpr int CHUNK_SIZE = 64;
  double totalSeconds = 0.0;
  int remaining = iters;
  while(remaining > 0) {
    int chunk = std::min(remaining, CHUNK_SIZE);
    remaining -= chunk;
    if(useGpuTimestamps) {
      beginRecording();
      vkCmdResetQueryPool(cmd, queryPool, 0, 2);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
      for(int i = 0; i < chunk; i++)
        recordOne();
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
      submitAndWait();
      uint64_t ts[2] = {0, 0};
      VkResult r = vkGetQueryPoolResults(
        device(), queryPool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      if(r != VK_SUCCESS || ts[1] <= ts[0])
        return -1.0;
      totalSeconds += (double)(ts[1] - ts[0]) * timestampPeriodNs * 1e-9;
    } else {
      beginRecording();
      for(int i = 0; i < chunk; i++)
        recordOne();
      auto t0 = std::chrono::steady_clock::now();
      submitAndWait();
      auto t1 = std::chrono::steady_clock::now();
      totalSeconds += std::chrono::duration<double>(t1 - t0).count();
    }
  }
  return totalSeconds;
}

void VulkanWorkContext::warmupDispatches(
  int chunkSize,
  int numChunks,
  const std::function<void()>& recordOne) {
  if(chunkSize <= 0 || numChunks <= 0)
    return;

  // Each chunk is a separate submission. This is intentionally untimed: it
  // settles cache/pipeline state immediately before the measured probe starts.
  for(int chunk = 0; chunk < numChunks; chunk++) {
    beginRecording();
    for(int i = 0; i < chunkSize; i++)
      recordOne();
    submitAndWait();
  }
}

// VulkanWorkContext accessors that need the full CompiledPipelines layout (defined
// in this TU). Bodies are tiny; declared in vulkanbackend.h.
bool VulkanWorkContext::resolvedFP16Storage() const {
  return pipelines->usingFP16Storage;
}
bool VulkanWorkContext::resolvedFP16Compute() const {
  return pipelines->usingFP16Compute;
}

// VulkanTuner::TuningContext FP16-aware upload/download. Lives next to
// VulkanWorkContext in vulkanbackend.h; implementations here use the same
// VulkanHelpers utilities the rest of this TU does.
VBuf VulkanTuner::TuningContext::makeInputBufFP(const std::vector<float>& data) {
  if(!fp16Storage)
    return makeInputBuf(data);
  auto buf = makeDeviceBuf(device(), memProps(), std::max<size_t>(data.size(), 1), true);
  VkDeviceSize byteSize = data.size() * sizeof(half_t);
  if(byteSize == 0)
    return buf;
  auto staging = VulkanBuffer(allocateBuffer(
    device(),
    memProps(),
    byteSize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  void* mapped;
  VK_CHECK(vkMapMemory(device(), staging.memory, 0, byteSize, 0, &mapped));
  half_t* dst = static_cast<half_t*>(mapped);
  for(size_t i = 0; i < data.size(); i++)
    dst[i] = half_float::half_cast<half_t>(data[i]);
  vkUnmapMemory(device(), staging.memory);
  submitCopy(staging.buffer, buf->buffer, byteSize);
  return buf;
}

void VulkanTuner::TuningContext::downloadFloatsFP(VulkanBuffer* srcBuf, std::vector<float>& dst, size_t numElts) {
  if(!fp16Storage) {
    downloadFloats(srcBuf, dst, numElts);
    return;
  }
  VkDeviceSize stagingSize = numElts * sizeof(half_t);
  if(stagingSize == 0)
    return;
  auto staging = VulkanBuffer(allocateBuffer(
    device(),
    memProps(),
    stagingSize,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  beginRecording();
  VkBufferCopy r = {0, 0, stagingSize};
  vkCmdCopyBuffer(cmd, srcBuf->buffer, staging.buffer, 1, &r);
  submitAndWait();
  void* mapped;
  VK_CHECK(vkMapMemory(device(), staging.memory, 0, stagingSize, 0, &mapped));
  const half_t* src = static_cast<const half_t*>(mapped);
  for(size_t i = 0; i < numElts; i++)
    dst[i] = (float)src[i];
  vkUnmapMemory(device(), staging.memory);
}

// ============================================================================
// Section: Runtime Autotuning (ComputeContext::tuneSelf)
// ============================================================================
// ComputeContext::tuneSelf
//
// For each device whose tuneParams came from compiled-in defaults during ctor
// (i.e. cache miss), open a TuningContext borrowed against THIS ComputeContext
// and that device, run the sweep, and write the winning params into the
// device's CompiledPipelines::tuneParams. Mutating tuneParams is safe because
// CompiledPipelines does NOT pre-compile inference pipelines in its ctor —
// those are built lazily inside ComputeHandle/Model construction. We guard
// against the late-mutation case with handlesAlreadyCreated.

void ComputeContext::tuneSelf(
  const ModelDesc* modelDesc,
  int batchSize,
  int winograd3x3OutTile,
  int benchIters,
  bool verboseTuner,
  bool full,
  const std::string& saveFileOverride) {
  if(handlesAlreadyCreated)
    throw StringError(
      "ComputeContext::tuneSelf called after ComputeHandle creation - pipelines already compiled with stale spec "
      "constants");
  testAssert(modelDesc != nullptr);

  // When tuneSelf is invoked directly (e.g. by runTuneCommand) on a context that
  // was built without modelDescForTuneLoad, every owned device needs tuning.
  // When it's invoked by createComputeContext after a partial cache hit, only the
  // devices on gpuIdxsNeedingTune do.
  std::vector<int> toTune = gpuIdxsNeedingTune;
  if(toTune.empty()) {
    for(auto& dev: devicesContext->devicesToUse)
      toTune.push_back(dev->info.gpuIdx);
  }

  // Multiple gpuIdxs can resolve to the same physical device, and identical
  // GPUs on the same host share tuning (architecture-level parameters, not
  // per-die). Dedup by tuningDeviceKey — (vendorID, deviceID, driverVersion) —
  // so two RTX 4090s tune once and reuse the result.
  std::map<std::string, VulkanTuneParams> tunedParamsByKey;

  for(int gpuIdx: toTune) {
    const InitializedVulkanDevice* dev = devicesContext->findGpuExn(gpuIdx);
    const std::string gpuName(dev->info.properties.deviceName);
    const std::string tuneKey = tuningDeviceKey(dev);

    auto alreadyTuned = tunedParamsByKey.find(tuneKey);
    if(alreadyTuned != tunedParamsByKey.end()) {
      CompiledPipelines* pipelines = compiledPipelinesByDevice.at(dev->info.physicalDevice);
      pipelines->tuneParams = alreadyTuned->second;
      if(logger != nullptr)
        logger->write(
          "Reusing Vulkan tuning results for device " + Global::intToString(gpuIdx) +
          " (same tuning key as an earlier tuned GPU): " + gpuName);
      continue;
    }

    if(logger != nullptr) {
      logger->write("Performing Vulkan autotuning for " + gpuName + " for batch " + Global::intToString(batchSize));
      logger->write("*** On some systems, this may take several minutes, please be patient ***");
    }

    string outputPath;
    if(!saveFileOverride.empty()) {
      outputPath = saveFileOverride;
    } else {
      string dir = VulkanTuner::defaultDirectory(homeDataDirOverride);
      outputPath = dir + "/" +
                   VulkanTuner::defaultFileName(
                     gpuName,
                     tuneKey,
                     nnXLen,
                     nnYLen,
                     modelDesc,
                     compiledPipelinesByDevice.at(dev->info.physicalDevice)->usingFP16Storage);
    }

    CompiledPipelines* pipelines = compiledPipelinesByDevice.at(dev->info.physicalDevice);
    // Normal model startup already loaded and checked its cache in the
    // ComputeContext constructor. The standalone tuner cannot do that there,
    // because its model and optional output path are supplied only to tuneSelf.
    // Reuse a current, valid output file here so the tuner fills only kernel
    // families that are missing for the requested layout/model/filter.
    if(!autoTuneEnabled) {
      std::ifstream probe;
      const bool outputExists = FileUtils::tryOpen(probe, outputPath);
      if(outputExists) {
        try {
          VulkanTuneParams loaded = VulkanTuneParams::load(outputPath);
          // Single load-time validity gate; what remains tuned is valid and can
          // seed the incremental tune below.
          VulkanTuner::reconcileTuneParamsForDevice(
            loaded, dev->info, logger, pipelines->usingFP16Storage, pipelines->usingFP16Compute);
          pipelines->tuneParams = loaded;
          if(logger != nullptr)
            logger->write("Loaded existing vulkan tuning parameters for incremental tuning from: " + outputPath);
        } catch(const IOError& e) {
          if(logger != nullptr)
            logger->write(
              "Vulkan tune file could not be loaded, ignoring and re-tuning: " + outputPath + " (" + e.what() + ")");
        }
      }
    }

    VulkanTuner::TuningContext session;
    session.logger = logger;
    session.initBorrowed(this, gpuIdx, logger);
    session.fp16Storage = session.resolvedFP16Storage();
    session.fp16Compute = session.resolvedFP16Compute();

    VulkanTuneParams initialParams = pipelines->tuneParams;
    if(initialParams.nhwcWinograd3x3OutTile != winograd3x3OutTile) {
      if(logger != nullptr)
        logger->write(
          "Vulkan Winograd output tile changed from " + Global::intToString(initialParams.nhwcWinograd3x3OutTile) +
          " to " + Global::intToString(winograd3x3OutTile) + "; invalidating Winograd tuning.");
      initialParams.clearKernelTuned(
        {VulkanTuner::TUNED_CONV3X3_WINOGRAD,
         VulkanTuner::TUNED_CONV5X5_WINOGRAD,
         VulkanTuner::TUNED_WINOGRAD_TRANSFORM,
         VulkanTuner::TUNED_WINOGRAD_UNTRANSFORM});
      initialParams.nhwcWinogradTransformTunedLayout.clear();
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_DOT2});
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16});
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_COOPMAT1});
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16});
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_COOPMAT2});
      initialParams.clearKernelTuned({VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16});
    }
    initialParams.nhwcWinograd3x3OutTile = winograd3x3OutTile;
    nhwcWinograd3x3OutTileFor(initialParams);

    VulkanTuneParams results;
    VulkanTuner::tune(
      &session,
      initialParams,
      modelDesc,
      batchSize,
      nnXLen,
      nnYLen,
      benchIters,
      std::cerr,
      verboseTuner,
      full,
      results);
    session.destroy();

    // Write the winners back into THIS device's CompiledPipelines so subsequent
    // ComputeHandle creation picks them up.
    pipelines->tuneParams = results;
    tunedParamsByKey[tuneKey] = results;

    VulkanTuneParams::save(outputPath, results);
    if(logger != nullptr) {
      std::ostringstream s;
      VulkanTuner::appendTuneParamsSummary(s, results, modelDesc);
      logger->write(
        "Using vulkan tuning parameters for GPU " + Global::intToString(gpuIdx) + " (" + gpuName + "): " + s.str());
      logger->write("Done Vulkan tuning, saved results to " + outputPath);
    }
  }

  gpuIdxsNeedingTune.clear();
}

// ============================================================================
// Section: Inference Runtime Structures (Buffers/Model/Handle)
// ============================================================================
// Buffers: per-ComputeHandle persistent GPU buffers

struct Buffers {
  VkDevice device;
  bool fp16;
  int maxBatchSize;  // used to scale per-batch copy sizes from the max-batch *Elts fields below
  int inputChannels;
  size_t inputElts, inputGlobalElts, inputMetaElts;
  size_t maskElts, maskSumElts;
  size_t policyElts, policyPassElts;
  size_t valueElts, scoreValueElts, ownershipElts;
  std::unique_ptr<PlannedBufferAllocator> persistentAllocator;

  // Byte offsets of each output tensor within stagingDownload.
  // Computed from the fixed max-batch sizes so readbackOutputs() can find data at
  // stable addresses regardless of which per-batch cmd buffer was submitted.
  static constexpr size_t align4(size_t x) { return (x + 3u) & ~(static_cast<size_t>(3u)); }
  size_t dlPolicyPassBytes() const { return policyPassElts * sizeof(float); }
  size_t dlPolicyBytes() const { return policyElts * (fp16 ? sizeof(half_t) : sizeof(float)); }
  size_t dlValueBytes() const { return valueElts * sizeof(float); }
  size_t dlScoreValueBytes() const { return scoreValueElts * sizeof(float); }
  size_t dlOwnershipBytes() const { return ownershipElts * (fp16 ? sizeof(half_t) : sizeof(float)); }
  size_t dlPolicyPassOffset() const { return 0; }
  size_t dlPolicyOffset() const { return align4(dlPolicyPassOffset() + dlPolicyPassBytes()); }
  size_t dlValueOffset() const { return align4(dlPolicyOffset() + dlPolicyBytes()); }
  size_t dlScoreValueOffset() const { return align4(dlValueOffset() + dlValueBytes()); }
  size_t dlOwnershipOffset() const { return align4(dlScoreValueOffset() + dlScoreValueBytes()); }

  // Scale a max-batch byte count down to b items. Each *Elts above is exactly
  // (per-item-elts * maxBatchSize), so dividing by maxBatchSize and multiplying by b
  // gives the per-b byte count without storing redundant per-item fields.
  size_t scaleBytes(size_t maxBatchBytes, int b) const { return maxBatchBytes / (size_t)maxBatchSize * (size_t)b; }

  VBuf input;
  VBuf inputGlobal;
  VBuf inputMeta;
  VBuf mask;
  VBuf maskSum;
  VBuf policy, policyPass;
  VBuf value, scoreValue, ownership;

  // Direct input path on integrated GPUs: host-visible mapped input buffers.
  bool usingDirectInputMapping;
  void* inputMapped;
  void* inputGlobalMapped;
  void* inputMetaMapped;

  // Staging buffers (HOST_VISIBLE+COHERENT) for upload/download
  VBuf stagingUploadSpatial;  // for input
  VBuf stagingUploadGlobal;
  VBuf stagingUploadMeta;
  VBuf stagingUploadMask;
  VBuf stagingUploadMaskSum;
  VBuf stagingDownload;  // for all outputs
  void* stagingUploadSpatialMapped;
  void* stagingUploadGlobalMapped;
  void* stagingUploadMetaMapped;
  void* stagingUploadMaskMapped;
  void* stagingUploadMaskSumMapped;
  void* stagingDownloadMapped;

  Buffers(
    VkDevice dev,
    const VkPhysicalDeviceMemoryProperties& memProps,
    const ModelDesc& model,
    int maxBatchSize_,
    int paddedSpatialSize,
    int nnXYLen,
    int inputChannels_,
    bool fp16_param,
    bool tryDirectInputMapping)
    : device(dev),
      fp16(fp16_param),
      maxBatchSize(maxBatchSize_),
      inputChannels(inputChannels_),
      usingDirectInputMapping(false),
      inputMapped(nullptr),
      inputGlobalMapped(nullptr),
      inputMetaMapped(nullptr),
      stagingUploadSpatialMapped(nullptr),
      stagingUploadGlobalMapped(nullptr),
      stagingUploadMetaMapped(nullptr),
      stagingUploadMaskMapped(nullptr),
      stagingUploadMaskSumMapped(nullptr),
      stagingDownloadMapped(nullptr) {
    int B = maxBatchSize;
    testAssert(inputChannels >= model.numInputChannels);

    inputElts = static_cast<size_t>(inputChannels) * B * paddedSpatialSize;
    inputGlobalElts = static_cast<size_t>(model.numInputGlobalChannels) * B;
    inputMetaElts = static_cast<size_t>(model.numInputMetaChannels) * B;
    maskElts = static_cast<size_t>(B) * paddedSpatialSize;
    maskSumElts = static_cast<size_t>(B);
    policyElts = static_cast<size_t>(model.numPolicyChannels) * B * paddedSpatialSize;
    policyPassElts = static_cast<size_t>(model.numPolicyChannels) * B;
    valueElts = static_cast<size_t>(model.numValueChannels) * B;
    scoreValueElts = static_cast<size_t>(model.numScoreValueChannels) * B;
    ownershipElts = static_cast<size_t>(model.numOwnershipChannels) * B * paddedSpatialSize;

    const VkBufferUsageFlags persistentUsage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto persistentBytes = [&](size_t elts, bool isFP16) -> VkDeviceSize {
      return (VkDeviceSize)(elts * (isFP16 ? sizeof(half_t) : sizeof(float)));
    };
    auto allocatePersistentDedicated = [&]() {
      if(!usingDirectInputMapping) {
        input = makeDeviceBuf(dev, memProps, inputElts, fp16);
        inputGlobal = makeDeviceBufFP32(dev, memProps, inputGlobalElts);
        if(inputMetaElts > 0)
          inputMeta = makeDeviceBufFP32(dev, memProps, inputMetaElts);
      }
      mask = makeDeviceBuf(dev, memProps, maskElts, fp16);
      maskSum = makeDeviceBufFP32(dev, memProps, maskSumElts);
      policy = makeDeviceBuf(dev, memProps, policyElts, fp16);
      policyPass = makeDeviceBufFP32(dev, memProps, policyPassElts);
      value = makeDeviceBufFP32(dev, memProps, valueElts);
      scoreValue = makeDeviceBufFP32(dev, memProps, scoreValueElts);
      ownership = makeDeviceBuf(dev, memProps, ownershipElts, fp16);
    };

    // Buffers common to both allocation paths (direct-mapped and staged). Input
    // trio is added separately by the staged path since the direct-mapped path
    // has already carved out DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT memory for
    // them above.
    auto addSharedPersistent = [&]() {
      persistentAllocator->add(
        &mask, persistentBytes(maskElts, fp16), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &maskSum, persistentBytes(maskSumElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &policy, persistentBytes(policyElts, fp16), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &policyPass, persistentBytes(policyPassElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &value, persistentBytes(valueElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &scoreValue, persistentBytes(scoreValueElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      persistentAllocator->add(
        &ownership, persistentBytes(ownershipElts, fp16), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    if(tryDirectInputMapping) {
      try {
        const VkMemoryPropertyFlags directInputMemFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        input = makeDeviceBuf(dev, memProps, inputElts, fp16, directInputMemFlags);
        inputGlobal = makeDeviceBufFP32(dev, memProps, inputGlobalElts, directInputMemFlags);
        if(inputMetaElts > 0)
          inputMeta = makeDeviceBufFP32(dev, memProps, inputMetaElts, directInputMemFlags);

        VK_CHECK(vkMapMemory(dev, input->memory, 0, input->size, 0, &inputMapped));
        VK_CHECK(vkMapMemory(dev, inputGlobal->memory, 0, inputGlobal->size, 0, &inputGlobalMapped));
        if(inputMetaElts > 0)
          VK_CHECK(vkMapMemory(dev, inputMeta->memory, 0, inputMeta->size, 0, &inputMetaMapped));
        usingDirectInputMapping = true;
      } catch(const StringError&) {
        input.reset();
        inputGlobal.reset();
        inputMeta.reset();
        inputMapped = nullptr;
        inputGlobalMapped = nullptr;
        inputMetaMapped = nullptr;
        usingDirectInputMapping = false;
      }
    }
    if(!usingDirectInputMapping) {
      try {
        persistentAllocator = std::make_unique<PlannedBufferAllocator>(dev, memProps);
        persistentAllocator->add(
          &input, persistentBytes(inputElts, fp16), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        persistentAllocator->add(
          &inputGlobal, persistentBytes(inputGlobalElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(inputMetaElts > 0)
          persistentAllocator->add(
            &inputMeta, persistentBytes(inputMetaElts, false), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        addSharedPersistent();
        persistentAllocator->build();
      } catch(const StringError&) {
        persistentAllocator.reset();
        allocatePersistentDedicated();
      }
    } else {
      try {
        persistentAllocator = std::make_unique<PlannedBufferAllocator>(dev, memProps);
        addSharedPersistent();
        persistentAllocator->build();
      } catch(const StringError&) {
        persistentAllocator.reset();
        allocatePersistentDedicated();
      }
    }

    // Staging: sized for worst case
    size_t maxUploadBytes = inputElts * (fp16 ? sizeof(half_t) : sizeof(float));
    stagingUploadSpatial = std::make_unique<VulkanBuffer>(allocateBuffer(
      dev,
      memProps,
      maxUploadBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(
      vkMapMemory(dev, stagingUploadSpatial->memory, 0, stagingUploadSpatial->size, 0, &stagingUploadSpatialMapped));
    stagingUploadGlobal = std::make_unique<VulkanBuffer>(allocateBuffer(
      dev,
      memProps,
      inputGlobalElts * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(
      vkMapMemory(dev, stagingUploadGlobal->memory, 0, stagingUploadGlobal->size, 0, &stagingUploadGlobalMapped));
    if(inputMetaElts > 0) {
      stagingUploadMeta = std::make_unique<VulkanBuffer>(allocateBuffer(
        dev,
        memProps,
        inputMetaElts * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
      VK_CHECK(vkMapMemory(dev, stagingUploadMeta->memory, 0, stagingUploadMeta->size, 0, &stagingUploadMetaMapped));
    }
    stagingUploadMask = std::make_unique<VulkanBuffer>(allocateBuffer(
      dev,
      memProps,
      maskElts * (fp16 ? sizeof(half_t) : sizeof(float)),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(vkMapMemory(dev, stagingUploadMask->memory, 0, stagingUploadMask->size, 0, &stagingUploadMaskMapped));
    if(!fp16) {
      float* maskData = static_cast<float*>(stagingUploadMaskMapped);
      memset(maskData, 0, stagingUploadMask->size);
      for(int n = 0; n < B; n++) {
        float* row = maskData + static_cast<size_t>(n) * paddedSpatialSize;
        std::fill(row, row + nnXYLen, 1.0f);
      }
    } else {
      half_t* maskData = static_cast<half_t*>(stagingUploadMaskMapped);
      const half_t zero = half_float::half_cast<half_t>(0.0f);
      const half_t one = half_float::half_cast<half_t>(1.0f);
      std::fill(maskData, maskData + maskElts, zero);
      for(int n = 0; n < B; n++) {
        half_t* row = maskData + static_cast<size_t>(n) * paddedSpatialSize;
        std::fill(row, row + nnXYLen, one);
      }
    }

    stagingUploadMaskSum = std::make_unique<VulkanBuffer>(allocateBuffer(
      dev,
      memProps,
      maskSumElts * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(
      vkMapMemory(dev, stagingUploadMaskSum->memory, 0, stagingUploadMaskSum->size, 0, &stagingUploadMaskSumMapped));
    {
      float* maskSumData = static_cast<float*>(stagingUploadMaskSumMapped);
      for(int n = 0; n < B; n++)
        maskSumData[n] = (float)nnXYLen;
    }

    const size_t policyBytesMax = policyElts * (fp16 ? sizeof(half_t) : sizeof(float));
    const size_t policyPassBytesMax = policyPassElts * sizeof(float);
    const size_t valueBytesMax = valueElts * sizeof(float);
    const size_t scoreValueBytesMax = scoreValueElts * sizeof(float);
    const size_t ownershipBytesMax = ownershipElts * (fp16 ? sizeof(half_t) : sizeof(float));
    size_t maxDownloadBytes = 0;
    maxDownloadBytes = Buffers::align4(maxDownloadBytes + policyPassBytesMax);
    maxDownloadBytes = Buffers::align4(maxDownloadBytes + policyBytesMax);
    maxDownloadBytes = Buffers::align4(maxDownloadBytes + valueBytesMax);
    maxDownloadBytes = Buffers::align4(maxDownloadBytes + scoreValueBytesMax);
    maxDownloadBytes += ownershipBytesMax;
    stagingDownload = std::make_unique<VulkanBuffer>(allocateBuffer(
      dev,
      memProps,
      maxDownloadBytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(vkMapMemory(dev, stagingDownload->memory, 0, stagingDownload->size, 0, &stagingDownloadMapped));
  }

  ~Buffers() {
    if(inputMetaMapped != nullptr && inputMeta)
      vkUnmapMemory(device, inputMeta->memory);
    if(inputGlobalMapped != nullptr && inputGlobal)
      vkUnmapMemory(device, inputGlobal->memory);
    if(inputMapped != nullptr && input)
      vkUnmapMemory(device, input->memory);
    if(stagingDownloadMapped != nullptr && stagingDownload)
      vkUnmapMemory(device, stagingDownload->memory);
    if(stagingUploadMetaMapped != nullptr && stagingUploadMeta)
      vkUnmapMemory(device, stagingUploadMeta->memory);
    if(stagingUploadMaskSumMapped != nullptr && stagingUploadMaskSum)
      vkUnmapMemory(device, stagingUploadMaskSum->memory);
    if(stagingUploadMaskMapped != nullptr && stagingUploadMask)
      vkUnmapMemory(device, stagingUploadMask->memory);
    if(stagingUploadGlobalMapped != nullptr && stagingUploadGlobal)
      vkUnmapMemory(device, stagingUploadGlobal->memory);
    if(stagingUploadSpatialMapped != nullptr && stagingUploadSpatial)
      vkUnmapMemory(device, stagingUploadSpatial->memory);
  }

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
};

// Model: orchestrates trunk + policy head + value head

struct Model {
  const int modelVersion;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  int trunkNumChannels;
  const int paddedSpatialSize;
  const bool externalInputsUseNhwc;
  int numInputChannelsPhysical;

  std::unique_ptr<TrunkVk> trunk;
  std::unique_ptr<PolicyHeadVk> policyHead;
  std::unique_ptr<ValueHeadVk> valueHead;
  // Owned mask-setup kernels, dispatched by setupMask() from the eval path
  // (outside the layer tree): extractChannel0 derives the spatial mask from the
  // input, sumMaskSpatial reduces it to a per-batch maskSum.
  VkDevice device;
  ComputeKernel extractChannel0NhwcKernel;
  VulkanKernels::NchwToNhwc::Kernels nchwToNhwcKernel;
  ComputeKernel sumMaskSpatialKernel;

  Model(
    const VulkanLayerContext& ctx,
    const ModelDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16,
    bool externalInputsUseNhwc_)
    : modelVersion(desc->modelVersion),
      numInputChannels(desc->numInputChannels),
      numInputGlobalChannels(desc->numInputGlobalChannels),
      numInputMetaChannels(desc->numInputMetaChannels),
      numPolicyChannels(desc->numPolicyChannels),
      numValueChannels(desc->numValueChannels),
      numScoreValueChannels(desc->numScoreValueChannels),
      numOwnershipChannels(desc->numOwnershipChannels),
      trunkNumChannels(desc->trunk.trunkNumChannels),
      paddedSpatialSize(paddedSpatialSize_),
      externalInputsUseNhwc(externalInputsUseNhwc_),
      numInputChannelsPhysical(desc->numInputChannels),
      device(ctx.device) {
    trunk = std::make_unique<TrunkVk>(ctx, &desc->trunk, nnX, nnY, paddedSpatialSize_, useFP16);
    numInputChannelsPhysical = trunk->initialConv->inChannelsPhysical;
    testAssert(numInputChannelsPhysical >= numInputChannels);
    policyHead = std::make_unique<PolicyHeadVk>(ctx, &desc->policyHead, nnX, nnY, paddedSpatialSize_, useFP16);
    valueHead = std::make_unique<ValueHeadVk>(ctx, &desc->valueHead, nnX, nnY, paddedSpatialSize_, useFP16);
    extractChannel0NhwcKernel = VulkanKernels::ExtractChannel0Nhwc::build(ctx.device, ctx.pipelineCache, useFP16);
    // The graph is NHWC. Retain this conversion pipeline solely for the public
    // NCHW input ABI; native NHWC inputs bypass it completely.
    if(!externalInputsUseNhwc)
      nchwToNhwcKernel = VulkanKernels::NchwToNhwc::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams);
    sumMaskSpatialKernel = VulkanKernels::SumMaskSpatial::build(ctx.device, ctx.pipelineCache, useFP16);
  }

  ~Model() {
    extractChannel0NhwcKernel.destroy(device);
    if(!externalInputsUseNhwc)
      nchwToNhwcKernel.destroy(device);
    sumMaskSpatialKernel.destroy(device);
  }

  // Derive mask (channel 0 of input) and maskSum (per-batch spatial sum) for the
  // non-exact-mask eval path. Issues its own output barriers.
  void setupMask(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int numSpatialFeatures,
    int paddedSpatialSize_,
    int maxBatchSize) const {
    {
      testAssert(numSpatialFeatures == numInputChannels);
      if(externalInputsUseNhwc) {
        const int inputChannels = numInputChannelsPhysical;
        VulkanKernels::ExtractChannel0Nhwc::PC pc = {inputChannels, paddedSpatialSize_};
        VulkanKernels::ExtractChannel0Nhwc::dispatch(ctx, extractChannel0NhwcKernel, input, mask, pc, maxBatchSize);
      } else {
        VulkanBuffer* nhwcInput = input;
        std::unique_ptr<SizedBuf<VulkanBuffer*>> convertedInput;
        convertedInput = std::make_unique<SizedBuf<VulkanBuffer*>>(
          scratch->allocator.get(), scratch->getBufSizeXY(numInputChannelsPhysical));
        VulkanKernels::NchwToNhwc::PC transposePc = {
          numSpatialFeatures, numInputChannelsPhysical, paddedSpatialSize_, maxBatchSize};
        VulkanKernels::NchwToNhwc::dispatch(ctx, nchwToNhwcKernel, input, convertedInput->buf, transposePc);
        cmdComputeBarrier(ctx.cmd, convertedInput->buf->buffer);
        nhwcInput = convertedInput->buf;
        VulkanKernels::ExtractChannel0Nhwc::PC pc = {numInputChannelsPhysical, paddedSpatialSize_};
        VulkanKernels::ExtractChannel0Nhwc::dispatch(ctx, extractChannel0NhwcKernel, nhwcInput, mask, pc, maxBatchSize);
      }
      cmdComputeBarrier(ctx.cmd, mask->buffer);
      // mark output mask of extractChannel0 input dependence for sumMaskSpatial and the BN/gpool consumers
    }
    {
      VulkanKernels::SumMaskSpatial::PC pc = {paddedSpatialSize_, maxBatchSize};
      VulkanKernels::SumMaskSpatial::dispatch(ctx, sumMaskSpatialKernel, mask, maskSum, pc, maxBatchSize);
      cmdComputeBarrier(ctx.cmd, maskSum->buffer);
      // mark output maskSum of sumMaskSpatial input dependence for gpool/valueHeadPool/spatialRMSNorm
    }
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    int maxBatchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership) const {
    // Spatial layout is owned by the model, rather than by individual layers.
    // Only the public NCHW boundaries get a conversion.
    SizedBuf<VulkanBuffer*> convertedInput(scratch->allocator.get(), scratch->getBufSizeXY(numInputChannelsPhysical));
    VulkanBuffer* nativeInput = input;
    if(!externalInputsUseNhwc) {
      VulkanKernels::NchwToNhwc::PC pc = {numInputChannels, numInputChannelsPhysical, paddedSpatialSize, maxBatchSize};
      VulkanKernels::NchwToNhwc::dispatch(ctx, nchwToNhwcKernel, input, convertedInput.buf, pc);
      cmdComputeBarrier(ctx.cmd, convertedInput.buf->buffer);
      nativeInput = convertedInput.buf;
    }
    SizedBuf<VulkanBuffer*> nativeTrunk(scratch->allocator.get(), scratch->getBufSizeXY(trunkNumChannels));
    trunk->dispatch(ctx, scratch, nativeInput, inputGlobal, inputMeta, nativeTrunk.buf, mask, maskSum, maxBatchSize);
    // trunk->dispatch barriers its own trunk tip output, so the heads read it directly.
    policyHead->dispatch(ctx, scratch, nativeTrunk.buf, mask, maskSum, policyPass, policy, maxBatchSize);
    valueHead->dispatch(ctx, scratch, nativeTrunk.buf, mask, maskSum, value, scoreValue, ownership, maxBatchSize);
    // policy/policyPass/value/scoreValue/ownership output barriers (compute->transfer
    // for download) are issued by the caller (VulkanOutputRun::recordMaskAndInference).
  }

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
    vector<VkDeviceSize> slots = trunk->permanentScratchSlotSizes(maxBatchSize, elemBytes);
    mergePermanentScratchSlots(slots, policyHead->permanentScratchSlotSizes(maxBatchSize, elemBytes));
    mergePermanentScratchSlots(slots, valueHead->permanentScratchSlotSizes(maxBatchSize, elemBytes));
    return slots;
  }
};

// InputBuffers (host-side only)

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  std::unique_ptr<float[]> userInputBuffer;
  std::unique_ptr<float[]> userInputGlobalBuffer;
  std::unique_ptr<float[]> userInputMetaBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;

    singleInputElts = static_cast<size_t>(m.numInputChannels) * nnXLen * nnYLen;
    singleInputGlobalElts = static_cast<size_t>(m.numInputGlobalChannels);
    singleInputMetaElts = static_cast<size_t>(m.numInputMetaChannels);

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);

    size_t userInputBufferElts = static_cast<size_t>(m.numInputChannels) * maxBatchSize * nnXLen * nnYLen;
    size_t userInputGlobalBufferElts = static_cast<size_t>(m.numInputGlobalChannels) * maxBatchSize;
    size_t userInputMetaBufferElts = static_cast<size_t>(m.numInputMetaChannels) * maxBatchSize;

    userInputBuffer = std::make_unique<float[]>(userInputBufferElts);
    userInputGlobalBuffer = std::make_unique<float[]>(userInputGlobalBufferElts);
    userInputMetaBuffer = std::make_unique<float[]>(userInputMetaBufferElts);
  }

  ~InputBuffers() = default;

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

// ComputeHandle

struct ComputeHandle {
  // Vulkan per-thread GPU resources
  ComputeContext* computeContext;
  VkDevice device;
  VkQueue queue;
  mutex* queueMutex;
  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> perBatchCmd;  // index 0 unused; [b] = cmd for live batch b
  VkFence fence;
  bool usingFP16Storage;
  bool usingFP16Compute;
  int paddedSpatialSize;
  // Cached for DispatchProfiler: device's timestamp granularity (ns/tick) and
  // whether the compute queue can record timestamps at all (validBits > 0).
  double timestampPeriodNs = 1.0;
  bool computeQueueSupportsTimestamps = false;
  CompiledPipelines* pipelines;
  PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

  // Model and buffers
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  int nnXLen, nnYLen;
  int maxBatchSize;
  bool inputsUseNHWC;
  bool requireExactNNLen;
  bool profileBatchSizes;
  std::vector<uint64_t> batchSizeCounts;

  // Per-dispatch GPU-timestamp profiler. Active iff KATAGO_VULKAN_PROFILE_KERNELS=1
  // and the device's compute queue supports timestamps. The reset and timestamp
  // writes are baked into the pre-recorded command buffer at construction time;
  // readback() + printReport() run per submit after vkWaitForFences. Also carries
  // the KATAGO_VULKAN_TRACE_KERNELS name-only trace mode (no query pool needed).
  VulkanProfiler::DispatchProfiler profiler;

  ComputeHandle(
    ComputeContext* ctx,
    const LoadedModel* lm,
    int maxBatchSize_,
    int gpuIdx,
    bool inputsUseNHWC_,
    bool requireExactNNLen_)
    : computeContext(ctx),
      nnXLen(ctx->nnXLen),
      nnYLen(ctx->nnYLen),
      maxBatchSize(maxBatchSize_),
      inputsUseNHWC(inputsUseNHWC_),
      requireExactNNLen(requireExactNNLen_),
      profileBatchSizes(vulkanBatchProfileEnabled()) {
    const InitializedVulkanDevice* dev = ctx->devicesContext->findGpuExn(gpuIdx);
    device = dev->device;
    InitializedVulkanDevice::QueueContext computeCtx = dev->pickComputeContext();
    queue = computeCtx.queue;
    queueMutex = computeCtx.mutex;

    // Dedicated command pool per handle to avoid cross-thread command-pool races.
    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpCI.queueFamilyIndex = (uint32_t)dev->info.computeQueueFamilyIdx;
    VK_CHECK(vkCreateCommandPool(device, &cpCI, nullptr, &commandPool));

    pipelines = ctx->compiledPipelinesByDevice[dev->info.physicalDevice];
    testAssert(pipelines != nullptr);
    paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
    usingFP16Storage = pipelines->usingFP16Storage;
    usingFP16Compute = pipelines->usingFP16Compute;

    timestampPeriodNs = dev->info.properties.limits.timestampPeriod;
    {
      uint32_t qfCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(dev->info.physicalDevice, &qfCount, nullptr);
      std::vector<VkQueueFamilyProperties> qfProps(qfCount);
      vkGetPhysicalDeviceQueueFamilyProperties(dev->info.physicalDevice, &qfCount, qfProps.data());
      int cqf = dev->info.computeQueueFamilyIdx;
      uint32_t validBits = (cqf >= 0 && cqf < (int)qfCount) ? qfProps[cqf].timestampValidBits : 0;
      computeQueueSupportsTimestamps = (validBits > 0) && (timestampPeriodNs > 0.0);
    }

    // Allocate one command buffer per live batch size (1..maxBatchSize) plus a fence.
    // Index 0 is left as VK_NULL_HANDLE (unused).
    perBatchCmd.assign(maxBatchSize + 1, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo cbAI = {};
    cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAI.commandPool = commandPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = (uint32_t)maxBatchSize;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbAI, perBatchCmd.data() + 1));

    VkFenceCreateInfo fenceCI = {};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(device, &fenceCI, nullptr, &fence));

    vkCmdPushDescriptorSetKHR =
      reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));

    InitializedVulkanDevice::QueueContext transferCtx = dev->getTransferContext();
    WeightUploadBatch weightUploadBatch(
      dev->device, transferCtx.queue, *transferCtx.mutex, transferCtx.commandPool, dev->info.memoryProperties);
    WeightUploadBatchGuard guard(&weightUploadBatch);

    VulkanLayerContext layerCtx = VulkanLayerContext::fromDevice(
      dev->device,
      transferCtx.queue,
      *transferCtx.mutex,
      transferCtx.commandPool,
      dev->info,
      pipelines->tuneParams,
      pipelines->pipelineCache);
    model = std::make_unique<Model>(
      layerCtx, &lm->modelDesc, nnXLen, nnYLen, paddedSpatialSize, usingFP16Storage, inputsUseNHWC_);
    weightUploadBatch.finish();

    scratch = std::make_unique<ScratchBuffers>(
      dev->device, dev->info.memoryProperties, usingFP16Storage, paddedSpatialSize, maxBatchSize);

    size_t elemBytes = usingFP16Storage ? sizeof(half_t) : sizeof(float);
    scratch->setPermanentSlots(model->permanentScratchSlotSizes(maxBatchSize, elemBytes));

    buffers = std::make_unique<Buffers>(
      dev->device,
      dev->info.memoryProperties,
      lm->modelDesc,
      maxBatchSize,
      paddedSpatialSize,
      nnXLen * nnYLen,
      model->externalInputsUseNhwc ? model->numInputChannelsPhysical : model->numInputChannels,
      usingFP16Storage,
      dev->info.isIntegrated);

    if(profileBatchSizes)
      batchSizeCounts.assign(maxBatchSize + 1, 0);

    if(VulkanProfiler::DispatchProfiler::profileEnvEnabled() || VulkanProfiler::DispatchProfiler::traceEnvEnabled()) {
      // NHWC layout islands add two repacks around every spatial convolution.
      // 2048 pairs covers those extra dispatches on the deepest current networks.
      profiler.init(device, timestampPeriodNs, computeQueueSupportsTimestamps, 2048);
    }

    for(int b = 1; b <= maxBatchSize; b++)
      preRecordCommandBufferFor(b, perBatchCmd[b]);

    // Trace mode: the slots now hold the kernel set recorded into the pre-recorded
    // command buffer(s), so print it once here rather than after every submit.
    if(profiler.traceKernels)
      profiler.printTrace(nullptr);
  }

  void preRecordCommandBufferFor(int b, VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;  // NOT ONE_TIME_SUBMIT — the buffer is submitted repeatedly
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Bake the timestamp query-pool reset into the recorded buffer so each submit
    // starts with cleared queries. recordStart/recordEnd from the kernel dispatches
    // below append their vkCmdWriteTimestamp pairs into the same pre-recorded buffer.
    if(profiler.active())
      profiler.beginCmd(cmd);

    bool directInputMapping = buffers->usingDirectInputMapping;
    int numSpatialFeatures = model->numInputChannels;
    int numMetaFeatures = model->numInputMetaChannels;
    const bool fp16 = usingFP16Storage;
    const size_t elemBytes = fp16 ? sizeof(half_t) : sizeof(float);

    // --- Upload copies (sized for b items) ---
    if(!directInputMapping) {
      VkBufferCopy region = {0, 0, buffers->scaleBytes(buffers->inputElts * elemBytes, b)};
      vkCmdCopyBuffer(cmd, buffers->stagingUploadSpatial->buffer, buffers->input->buffer, 1, &region);
    }
    {
      VkBufferCopy region = {0, 0, buffers->scaleBytes(buffers->inputGlobalElts * sizeof(float), b)};
      if(!directInputMapping)
        vkCmdCopyBuffer(cmd, buffers->stagingUploadGlobal->buffer, buffers->inputGlobal->buffer, 1, &region);
    }
    if(numMetaFeatures > 0 && !directInputMapping) {
      VkBufferCopy region = {0, 0, buffers->scaleBytes(buffers->inputMetaElts * sizeof(float), b)};
      vkCmdCopyBuffer(cmd, buffers->stagingUploadMeta->buffer, buffers->inputMeta->buffer, 1, &region);
    }

    auto inputBarrierFn = directInputMapping ? cmdHostToComputeBarrier : cmdTransferToComputeBarrier;
    inputBarrierFn(cmd, buffers->input->buffer);
    inputBarrierFn(cmd, buffers->inputGlobal->buffer);
    if(numMetaFeatures > 0)
      inputBarrierFn(cmd, buffers->inputMeta->buffer);

    // --- Mask + inference (sized for b items) ---
    CmdCtx cctx{cmd, vkCmdPushDescriptorSetKHR, profiler.active() ? &profiler : nullptr};
    // Fast path: requireExactNNLen is static for this handle's lifetime, so we
    // commit to the path at pre-record time. The caller-supplied mask is verified
    // to be all-ones per batch item in copyInputsToMappedBuffers (see there);
    // stagingUploadMask is pre-filled with all-ones (fp16 or fp32, max-batch);
    // stagingUploadMaskSum with nnXYLen per slot. Copying only the b-item prefix
    // is safe because the staging buffers are sized for maxBatchSize.
    if(requireExactNNLen) {
      VkBufferCopy maskRegion = {0, 0, buffers->scaleBytes(buffers->maskElts * elemBytes, b)};
      vkCmdCopyBuffer(cmd, buffers->stagingUploadMask->buffer, buffers->mask->buffer, 1, &maskRegion);
      VkBufferCopy maskSumRegion = {0, 0, buffers->scaleBytes(buffers->maskSumElts * sizeof(float), b)};
      vkCmdCopyBuffer(cmd, buffers->stagingUploadMaskSum->buffer, buffers->maskSum->buffer, 1, &maskSumRegion);
      cmdTransferToComputeBarrier(cmd, buffers->mask->buffer);
      cmdTransferToComputeBarrier(cmd, buffers->maskSum->buffer);
    } else {
      model->setupMask(
        cctx,
        scratch.get(),
        buffers->input.get(),
        buffers->mask.get(),
        buffers->maskSum.get(),
        numSpatialFeatures,
        paddedSpatialSize,
        b);
    }

    model->dispatch(
      cctx,
      scratch.get(),
      b,
      buffers->input.get(),
      buffers->inputGlobal.get(),
      buffers->inputMeta.get(),
      buffers->mask.get(),
      buffers->maskSum.get(),
      buffers->policyPass.get(),
      buffers->policy.get(),
      buffers->value.get(),
      buffers->scoreValue.get(),
      buffers->ownership.get());

    cmdComputeToTransferBarrier(cmd, buffers->policyPass->buffer);
    cmdComputeToTransferBarrier(cmd, buffers->policy->buffer);
    cmdComputeToTransferBarrier(cmd, buffers->value->buffer);
    cmdComputeToTransferBarrier(cmd, buffers->scoreValue->buffer);
    cmdComputeToTransferBarrier(cmd, buffers->ownership->buffer);

    // --- Download copies (sized for b items, at fixed max-batch offsets) ---
    VkBufferCopy region = {};
    region = {0, buffers->dlPolicyPassOffset(), buffers->scaleBytes(buffers->dlPolicyPassBytes(), b)};
    vkCmdCopyBuffer(cmd, buffers->policyPass->buffer, buffers->stagingDownload->buffer, 1, &region);
    region = {0, buffers->dlPolicyOffset(), buffers->scaleBytes(buffers->dlPolicyBytes(), b)};
    vkCmdCopyBuffer(cmd, buffers->policy->buffer, buffers->stagingDownload->buffer, 1, &region);
    region = {0, buffers->dlValueOffset(), buffers->scaleBytes(buffers->dlValueBytes(), b)};
    vkCmdCopyBuffer(cmd, buffers->value->buffer, buffers->stagingDownload->buffer, 1, &region);
    region = {0, buffers->dlScoreValueOffset(), buffers->scaleBytes(buffers->dlScoreValueBytes(), b)};
    vkCmdCopyBuffer(cmd, buffers->scoreValue->buffer, buffers->stagingDownload->buffer, 1, &region);
    region = {0, buffers->dlOwnershipOffset(), buffers->scaleBytes(buffers->dlOwnershipBytes(), b)};
    vkCmdCopyBuffer(cmd, buffers->ownership->buffer, buffers->stagingDownload->buffer, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmd));
  }

  // Submit the pre-recorded command buffer for the given live batch size and block until done.
  // Profiler readback (no-op when disabled) runs after the fence is signaled.
  void submitAndWait(int b) {
    testAssert(b > 0 && b <= maxBatchSize);
    VK_CHECK(vkResetFences(device, 1, &fence));
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &perBatchCmd[b];
    {
      lock_guard<mutex> lock(*queueMutex);
      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    }
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    if(profiler.timingActive()) {
      profiler.readback();
      profiler.printReport(nullptr, b);
    }
    if(profileBatchSizes)
      batchSizeCounts[b]++;
  }

  void printBatchSizeDistribution() const {
    if(!profileBatchSizes || batchSizeCounts.empty())
      return;
    uint64_t totalSubmits = 0;
    uint64_t totalRows = 0;
    for(int b = 1; b <= maxBatchSize; b++) {
      uint64_t count = batchSizeCounts[b];
      totalSubmits += count;
      totalRows += count * (uint64_t)b;
    }
    if(totalSubmits == 0)
      return;

    double avgBatchSize = (double)totalRows / (double)totalSubmits;
    std::ostringstream oss;
    oss << "[vk-batch-prof] batch size distribution (maxBatchSize=" << maxBatchSize << ", submits=" << totalSubmits
        << ", rows=" << totalRows << ", avg=" << avgBatchSize << "):\n";
    for(int b = 1; b <= maxBatchSize; b++) {
      uint64_t count = batchSizeCounts[b];
      if(count == 0)
        continue;
      double pct = 100.0 * (double)count / (double)totalSubmits;
      oss << "  batch=" << b << " count=" << count << " pct=" << pct << "%\n";
    }
    static std::mutex printMutex;
    lock_guard<mutex> lock(printMutex);
    std::cerr << oss.str();
  }

  ~ComputeHandle() {
    printBatchSizeDistribution();
    profiler.destroy();
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, maxBatchSize, perBatchCmd.data() + 1);
    vkDestroyCommandPool(device, commandPool, nullptr);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

// ============================================================================
// Section: NeuralNet Runtime Interface
// ============================================================================
// NeuralNet interface implementations

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {
  context->markHandleCreated();
  if(logger != nullptr) {
    logger->write(
      "Vulkan backend thread " + Global::intToString(serverThreadIdx) + ": model version " +
      Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write(
      "Vulkan backend thread " + Global::intToString(serverThreadIdx) + ": model " + loadedModel->modelDesc.name);
  }

  auto* h =
    new ComputeHandle(context, loadedModel, maxBatchSize, gpuIdxForThisThread, inputsUseNHWC, requireExactNNLen);

  if(logger != nullptr) {
    logger->write(
      "Vulkan backend thread " + Global::intToString(serverThreadIdx) + " FP16Storage " +
      Global::boolToString(h->usingFP16Storage) + " FP16Compute " + Global::boolToString(h->usingFP16Compute));
  }
  return h;
}

void NeuralNet::freeComputeHandle(ComputeHandle* h) {
  delete h;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* h) {
  // Only storage width is observable to callers. Some kernels may use FP16 ALU
  // internally when tuned, but the public useFP16 contract remains storage mode.
  return h->usingFP16Storage;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* h, bool isWarmup) {
  (void)h;
  (void)isWarmup;
  return false;
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* lm, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(lm, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* b) {
  delete b;
}

// getOutput: the main inference function — VulkanOutputRun orchestrates
// prepare → record → submit → readback → decode.

namespace {

  struct VulkanOutputRun {
    ComputeHandle* gpuHandle;
    InputBuffers* inputBuffers;
    NNResultBuf** inputBufs;
    vector<NNOutput*>& outputs;

    int batchSize;
    int nnXLen;
    int nnYLen;
    int nnXYLen;
    int modelVersion;
    int numSpatialFeatures;
    int numGlobalFeatures;
    int numMetaFeatures;
    int numPolicyChannels;

    ComputeHandle* handle;
    Buffers* bufs;
    int paddedSpatialSize;
    bool fp16;
    bool directInputMapping;

    vector<float> policyPassVec;
    vector<float> policyVec;
    vector<float> valueVec;
    vector<float> scoreValueVec;
    vector<float> ownershipVec;

    VulkanOutputRun(
      ComputeHandle* gpuHandle_,
      InputBuffers* inputBuffers_,
      int numBatchEltsFilled,
      NNResultBuf** inputBufs_,
      vector<NNOutput*>& outputs_)
      : gpuHandle(gpuHandle_),
        inputBuffers(inputBuffers_),
        inputBufs(inputBufs_),
        outputs(outputs_),
        batchSize(numBatchEltsFilled),
        nnXLen(gpuHandle_->nnXLen),
        nnYLen(gpuHandle_->nnYLen),
        nnXYLen(gpuHandle_->nnXLen * gpuHandle_->nnYLen),
        modelVersion(gpuHandle_->model->modelVersion),
        numSpatialFeatures(NNModelVersion::getNumSpatialFeatures(modelVersion)),
        numGlobalFeatures(NNModelVersion::getNumGlobalFeatures(modelVersion)),
        numMetaFeatures((int)inputBuffers_->singleInputMetaElts),
        numPolicyChannels(gpuHandle_->model->numPolicyChannels),
        handle(gpuHandle_),
        bufs(gpuHandle_->buffers.get()),
        paddedSpatialSize(handle->paddedSpatialSize),
        fp16(handle->usingFP16Storage),
        directInputMapping(bufs->usingDirectInputMapping) {}

    void prepareInputs() {
      for(int nIdx = 0; nIdx < batchSize; nIdx++) {
        float* rowSpatialInput = inputBuffers->userInputBuffer.get() + inputBuffers->singleInputElts * nIdx;
        float* rowGlobalInput = inputBuffers->userInputGlobalBuffer.get() + inputBuffers->singleInputGlobalElts * nIdx;

        const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
        const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();

        std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
        SymmetryHelpers::copyInputsWithSymmetry(
          rowSpatial,
          rowSpatialInput,
          1,
          nnYLen,
          nnXLen,
          numSpatialFeatures,
          gpuHandle->inputsUseNHWC,
          inputBufs[nIdx]->symmetry);

        if(numMetaFeatures > 0) {
          float* rowMetaInput = inputBuffers->userInputMetaBuffer.get() + inputBuffers->singleInputMetaElts * nIdx;
          testAssert(inputBufs[nIdx]->hasRowMeta);
          std::copy(
            inputBufs[nIdx]->rowMetaBuf.data(), inputBufs[nIdx]->rowMetaBuf.data() + numMetaFeatures, rowMetaInput);
        }
      }
      // Verify the mask is all-ones when requireExactNNLen is set. The fast
      // path uploads a pre-baked all-ones mask/maskSum regardless of what the
      // caller wrote, so silently accepting a non-1 mask would produce wrong
      // outputs. Check every batch item on every call — cheap next to the
      // inference cost and correctness matters more than the microseconds.
      if(gpuHandle->requireExactNNLen) {
        for(int nIdx = 0; nIdx < batchSize; nIdx++) {
          const float* maskRow = inputBuffers->userInputBuffer.get() + inputBuffers->singleInputElts * nIdx;
          for(int i = 0; i < nnXYLen; i++) {
            const int maskIdx = gpuHandle->inputsUseNHWC ? i * numSpatialFeatures : i;
            if(maskRow[maskIdx] != 1.0f)
              throw StringError(
                "Vulkan backend: requireExactNNLen=true but mask channel contains a "
                "non-1 value at batch " +
                Global::intToString(nIdx) + ", index " + Global::intToString(i) +
                " (value=" + Global::floatToString(maskRow[maskIdx]) +
                "). The exact-length fast path assumes all board positions are valid.");
          }
        }
      }
    }

    // Host-only: writes the symmetry-applied / padded inputs from the host scratch
    // (inputBuffers->userInput*Buffer) into the Vulkan-mapped memory that the
    // pre-recorded command buffer consumes — either the directly-mapped device
    // buffer (when direct input mapping is available) or the staging-upload buffer
    // (the pre-recorded vkCmdCopyBuffer / barrier sequence then carries it to the
    // device-local input buffer). Records no commands.
    void copyInputsToMappedBuffers() {
      const int uploadedSpatialChannels = bufs->inputChannels;
      const size_t spatialUploadBytes = static_cast<size_t>(uploadedSpatialChannels) * batchSize * paddedSpatialSize *
                                        (fp16 ? sizeof(half_t) : sizeof(float));
      {
        void* mapped = directInputMapping ? bufs->inputMapped : bufs->stagingUploadSpatialMapped;
        if(!fp16) {
          float* dst = static_cast<float*>(mapped);
          memset(dst, 0, spatialUploadBytes);
          for(int n = 0; n < batchSize; n++) {
            const float* src =
              inputBuffers->userInputBuffer.get() + n * static_cast<size_t>(numSpatialFeatures) * nnXYLen;
            if(gpuHandle->inputsUseNHWC) {
              for(int xy = 0; xy < nnXYLen; xy++)
                for(int c = 0; c < numSpatialFeatures; c++)
                  dst[(n * paddedSpatialSize + xy) * uploadedSpatialChannels + c] = src[xy * numSpatialFeatures + c];
            } else {
              testAssert(uploadedSpatialChannels == numSpatialFeatures);
              for(int c = 0; c < numSpatialFeatures; c++) {
                memcpy(
                  dst + (n * numSpatialFeatures + c) * paddedSpatialSize, src + c * nnXYLen, nnXYLen * sizeof(float));
              }
            }
          }
        } else {
          half_t* dst = static_cast<half_t*>(mapped);
          std::fill_n(dst, spatialUploadBytes / sizeof(half_t), half_float::half_cast<half_t>(0.0f));
          for(int n = 0; n < batchSize; n++) {
            const float* src =
              inputBuffers->userInputBuffer.get() + n * static_cast<size_t>(numSpatialFeatures) * nnXYLen;
            if(gpuHandle->inputsUseNHWC) {
              for(int xy = 0; xy < nnXYLen; xy++)
                for(int c = 0; c < numSpatialFeatures; c++)
                  dst[(n * paddedSpatialSize + xy) * uploadedSpatialChannels + c] =
                    half_float::half_cast<half_t>(src[xy * numSpatialFeatures + c]);
            } else {
              testAssert(uploadedSpatialChannels == numSpatialFeatures);
              for(int c = 0; c < numSpatialFeatures; c++) {
                half_t* dstRow = dst + (n * numSpatialFeatures + c) * paddedSpatialSize;
                const float* srcRow = src + c * nnXYLen;
                for(int xy = 0; xy < nnXYLen; xy++)
                  dstRow[xy] = half_float::half_cast<half_t>(srcRow[xy]);
              }
            }
          }
        }
      }
      {
        size_t globalBytes = static_cast<size_t>(numGlobalFeatures) * batchSize * sizeof(float);
        void* mapped = directInputMapping ? bufs->inputGlobalMapped : bufs->stagingUploadGlobalMapped;
        memcpy(mapped, inputBuffers->userInputGlobalBuffer.get(), globalBytes);
      }
      if(numMetaFeatures > 0) {
        size_t metaBytes = static_cast<size_t>(numMetaFeatures) * batchSize * sizeof(float);
        void* mapped = directInputMapping ? bufs->inputMetaMapped : bufs->stagingUploadMetaMapped;
        memcpy(mapped, inputBuffers->userInputMetaBuffer.get(), metaBytes);
      }
    }

    void readbackOutputs() {
      auto decodeF16ToF32 = [](const half_t* src, float* dst, size_t count) {
        for(size_t i = 0; i < count; i++)
          dst[i] = (float)src[i];
      };
      const int numPolicySpatialElts = numPolicyChannels * batchSize * paddedSpatialSize;
      const int numValueElts = gpuHandle->model->numValueChannels * batchSize;
      const int numScoreValueElts = gpuHandle->model->numScoreValueChannels * batchSize;
      const int numOwnershipElts = gpuHandle->model->numOwnershipChannels * batchSize * paddedSpatialSize;
      policyPassVec.resize(static_cast<size_t>(numPolicyChannels) * batchSize);
      policyVec.resize(static_cast<size_t>(numPolicySpatialElts));
      valueVec.resize(static_cast<size_t>(numValueElts));
      scoreValueVec.resize(static_cast<size_t>(numScoreValueElts));
      ownershipVec.resize(static_cast<size_t>(numOwnershipElts));

      // Use Buffers' download offset helpers — they derive offsets from the same
      // fixed max-batch sizes that preRecordCommandBuffer() used.
      const size_t pPassOff = bufs->dlPolicyPassOffset();
      const size_t polOff = bufs->dlPolicyOffset();
      const size_t valOff = bufs->dlValueOffset();
      const size_t svOff = bufs->dlScoreValueOffset();
      const size_t ownOff = bufs->dlOwnershipOffset();

      const size_t pPassBytes = static_cast<size_t>(numPolicyChannels) * batchSize * sizeof(float);
      const size_t polBytes = static_cast<size_t>(numPolicySpatialElts) * (fp16 ? sizeof(half_t) : sizeof(float));
      const size_t valBytes = static_cast<size_t>(numValueElts) * sizeof(float);
      const size_t svBytes = static_cast<size_t>(numScoreValueElts) * sizeof(float);
      const size_t ownBytes = static_cast<size_t>(numOwnershipElts) * (fp16 ? sizeof(half_t) : sizeof(float));

      void* mapped = bufs->stagingDownloadMapped;
      const char* base = static_cast<const char*>(mapped);
      memcpy(policyPassVec.data(), base + pPassOff, pPassBytes);
      if(!fp16)
        memcpy(policyVec.data(), base + polOff, polBytes);
      else
        decodeF16ToF32(reinterpret_cast<const half_t*>(base + polOff), policyVec.data(), policyVec.size());
      memcpy(valueVec.data(), base + valOff, valBytes);
      memcpy(scoreValueVec.data(), base + svOff, svBytes);
      if(!fp16)
        memcpy(ownershipVec.data(), base + ownOff, ownBytes);
      else
        decodeF16ToF32(reinterpret_cast<const half_t*>(base + ownOff), ownershipVec.data(), ownershipVec.size());
    }

    void writeOutputsToResults() {
      const int numValueChannels = gpuHandle->model->numValueChannels;
      const int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      const int numOwnershipChannels = gpuHandle->model->numOwnershipChannels;
      float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

      for(int nIdx = 0; nIdx < batchSize; nIdx++) {
        NNOutput* output = outputs[nIdx];
        assert(output->nnXLen == nnXLen);
        assert(output->nnYLen == nnYLen);
        float policyOptimism = (float)inputBufs[nIdx]->policyOptimism;

        // Value
        const float* v = &valueVec[nIdx * numValueChannels];
        output->whiteWinProb = v[0];
        output->whiteLossProb = (numValueChannels >= 3) ? v[1] : 0.0f;
        output->whiteNoResultProb = (numValueChannels >= 3) ? v[2] : 0.0f;

        // Score values
        const float* sv = &scoreValueVec[nIdx * numScoreValueChannels];
        if(modelVersion >= 9) {
          testAssert(numScoreValueChannels == 6);
          output->whiteScoreMean = sv[0];
          output->whiteScoreMeanSq = sv[1];
          output->whiteLead = sv[2];
          output->varTimeLeft = sv[3];
          output->shorttermWinlossError = sv[4];
          output->shorttermScoreError = sv[5];
        } else if(modelVersion >= 8) {
          testAssert(numScoreValueChannels == 4);
          output->whiteScoreMean = sv[0];
          output->whiteScoreMeanSq = sv[1];
          output->whiteLead = sv[2];
          output->varTimeLeft = sv[3];
          output->shorttermWinlossError = 0.0f;
          output->shorttermScoreError = 0.0f;
        } else if(modelVersion >= 4) {
          testAssert(numScoreValueChannels == 2);
          output->whiteScoreMean = sv[0];
          output->whiteScoreMeanSq = sv[1];
          output->whiteLead = sv[0];
          output->varTimeLeft = 0.0f;
          output->shorttermWinlossError = 0.0f;
          output->shorttermScoreError = 0.0f;
        } else if(modelVersion >= 3) {
          testAssert(numScoreValueChannels == 1);
          output->whiteScoreMean = sv[0];
          output->whiteScoreMeanSq = sv[0] * sv[0];
          output->whiteLead = sv[0];
          output->varTimeLeft = 0.0f;
          output->shorttermWinlossError = 0.0f;
          output->shorttermScoreError = 0.0f;
        } else {
          ASSERT_UNREACHABLE;
        }

        // Policy
        const int symmetry = inputBufs[nIdx]->symmetry;
        const float* policyPassSrc = &policyPassVec[nIdx * numPolicyChannels];
        const float* policySrc = &policyVec[nIdx * numPolicyChannels * paddedSpatialSize];
        if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
          auto blend = [policyOptimism](float p, float pOpt) { return p + (pOpt - p) * policyOptimism; };
          for(int i = 0; i < nnXYLen; i++)
            policyProbsTmp[i] = blend(policySrc[i * numPolicyChannels], policySrc[i * numPolicyChannels + 1]);
          output->policyProbs[nnXYLen] = blend(policyPassSrc[0], policyPassSrc[1]);
          SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, output->policyProbs, 1, nnYLen, nnXLen, symmetry);
        } else {
          assert(numPolicyChannels == 1);
          SymmetryHelpers::copyOutputsWithSymmetry(policySrc, output->policyProbs, 1, nnYLen, nnXLen, symmetry);
          output->policyProbs[nnXYLen] = policyPassSrc[0];
        }

        // Ownership
        if(output->whiteOwnerMap != nullptr) {
          testAssert(numOwnershipChannels == 1);
          const float* ow = &ownershipVec[nIdx * paddedSpatialSize];
          SymmetryHelpers::copyOutputsWithSymmetry(ow, output->whiteOwnerMap, 1, nnYLen, nnXLen, symmetry);
        }
      }
    }

    void run() {
      prepareInputs();
      copyInputsToMappedBuffers();
      handle->submitAndWait(batchSize);
      readbackOutputs();
      writeOutputsToResults();
    }
  };

}  // anonymous namespace

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  testAssert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  testAssert(numBatchEltsFilled > 0);
  VulkanOutputRun run(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
  run.run();
}

// ============================================================================
// Section: DispatchProfiler
// DispatchProfiler — KATAGO_VULKAN_PROFILE_KERNELS / KATAGO_VULKAN_TRACE_KERNELS
// instrumentation
// ============================================================================

bool VulkanProfiler::DispatchProfiler::profileEnvEnabled() {
  // Read the env var once per process. profileEnvEnabled() is called once in the
  // ComputeHandle constructor (before preRecordCommandBuffer) so the result
  // is baked into the pre-recorded command buffer.
  // getenv walks the process env each time — cache to avoid that on the hot path.
  static const bool cached = []() {
    std::string_view v = getenvStringView("KATAGO_VULKAN_PROFILE_KERNELS");
    return !v.empty() && v[0] != '0';
  }();
  return cached;
}

bool VulkanProfiler::DispatchProfiler::traceEnvEnabled() {
  static const bool cached = []() {
    std::string_view v = getenvStringView("KATAGO_VULKAN_TRACE_KERNELS");
    return !v.empty() && v[0] != '0';
  }();
  return cached;
}

void VulkanProfiler::DispatchProfiler::init(
  VkDevice device_,
  double timestampPeriodNs_,
  bool deviceSupportsTimestamps_,
  uint32_t capacityPairs_) {
  profileKernels = profileEnvEnabled();
  traceKernels = traceEnvEnabled();
  device = device_;
  timestampPeriodNs = timestampPeriodNs_;
  deviceSupportsTimestamps = deviceSupportsTimestamps_;
  capacityPairs = capacityPairs_;
  nextPair = 0;
  slots.reserve(capacityPairs);
  if(!profileKernels || !deviceSupportsTimestamps || capacityPairs == 0)
    return;
  VkQueryPoolCreateInfo qpCI = {};
  qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpCI.queryCount = capacityPairs * 2;
  VkResult r = vkCreateQueryPool(device, &qpCI, nullptr, &pool);
  if(r != VK_SUCCESS) {
    pool = VK_NULL_HANDLE;
    return;
  }
  // Print the banner once per process (not once per VulkanOutputRun — same process
  // may run many inferences and would otherwise spam stderr).
  static std::atomic<bool> bannerPrinted{false};
  bool expected = false;
  if(bannerPrinted.compare_exchange_strong(expected, true)) {
    std::cerr << "[vk-prof] DispatchProfiler enabled (capacity=" << capacityPairs
              << " pairs, timestampPeriod=" << timestampPeriodNs << "ns)" << std::endl;
  }
}

void VulkanProfiler::DispatchProfiler::destroy() {
  if(pool != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, pool, nullptr);
    pool = VK_NULL_HANDLE;
  }
  slots.clear();
  accums.clear();
  device = VK_NULL_HANDLE;
  profileKernels = false;
  traceKernels = false;
}

void VulkanProfiler::DispatchProfiler::beginCmd(VkCommandBuffer cmd) {
  if(!active())
    return;
  nextPair = 0;
  slots.clear();
  if(pool != VK_NULL_HANDLE)
    vkCmdResetQueryPool(cmd, pool, 0, capacityPairs * 2);
}

uint32_t VulkanProfiler::DispatchProfiler::recordStart(VkCommandBuffer cmd, std::string_view kernelName, bool fp16) {
  if(!active() || nextPair >= capacityPairs)
    return UINT32_MAX;
  uint32_t idx = nextPair++;
  slots.push_back({kernelName, fp16});
  if(pool != VK_NULL_HANDLE)
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, idx * 2);
  return idx;
}

void VulkanProfiler::DispatchProfiler::recordEnd(VkCommandBuffer cmd, uint32_t pairIdx) {
  if(pairIdx == UINT32_MAX || pool == VK_NULL_HANDLE)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, pairIdx * 2 + 1);
}

void VulkanProfiler::DispatchProfiler::readback() {
  accums.clear();
  if(!active() || nextPair == 0)
    return;
  std::vector<uint64_t> ts(nextPair * 2, 0);
  VkResult r = vkGetQueryPoolResults(
    device,
    pool,
    0,
    nextPair * 2,
    ts.size() * sizeof(uint64_t),
    ts.data(),
    sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if(r != VK_SUCCESS)
    return;
  std::map<std::string, DispatchProfiler::Accum> byName;
  for(uint32_t i = 0; i < nextPair; i++) {
    if(ts[2 * i + 1] <= ts[2 * i])
      continue;
    uint64_t delta = ts[2 * i + 1] - ts[2 * i];
    std::string key = std::string(slots[i].name) + (slots[i].fp16 ? "[fp16]" : "[fp32]");
    auto& a = byName[key];
    a.totalTicks += delta;
    a.count++;
  }
  accums.assign(byName.begin(), byName.end());
  std::sort(accums.begin(), accums.end(), [](const auto& a, const auto& b) {
    return a.second.totalTicks > b.second.totalTicks;
  });
}

void VulkanProfiler::DispatchProfiler::printReport(Logger* logger, int batchSize) const {
  if(accums.empty())
    return;
  uint64_t grandTotalTicks = 0;
  for(const auto& kv: accums)
    grandTotalTicks += kv.second.totalTicks;
  double grandTotalMs = (double)grandTotalTicks * timestampPeriodNs * 1e-6;
  std::ostringstream oss;
  oss << "[vk-prof] per-kernel breakdown (batch=" << batchSize << " total " << grandTotalMs << " ms over " << nextPair
      << " dispatches):\n";
  for(const auto& kv: accums) {
    double ms = (double)kv.second.totalTicks * timestampPeriodNs * 1e-6;
    double pct = grandTotalTicks > 0 ? 100.0 * (double)kv.second.totalTicks / (double)grandTotalTicks : 0.0;
    oss << "  " << kv.first << "  total=" << ms << "ms  pct=" << pct << "%  n=" << kv.second.count << "\n";
  }
  if(logger)
    logger->write(oss.str());
  else
    std::cerr << oss.str();
}

void VulkanProfiler::DispatchProfiler::printTrace(Logger* logger) const {
  if(!traceKernels || slots.empty())
    return;
  // Deduplicate while preserving first-seen order, and tally how many dispatches
  // hit each kernel. The slots here reflect the pre-recorded command buffer(s),
  // so this is the kernel set actually baked into the inference path.
  std::map<std::string, uint32_t> counts;
  std::vector<std::string> names;
  for(const auto& s: slots) {
    std::string key = std::string(s.name) + (s.fp16 ? "[fp16]" : "[fp32]");
    if(counts.emplace(key, 0).second)
      names.push_back(key);
    counts[key]++;
  }
  std::ostringstream oss;
  oss << "[vk-kernels] kernels in this pipeline (" << names.size() << " unique over " << slots.size()
      << " dispatches):\n";
  for(const auto& n: names)
    oss << "  " << n << "  n=" << counts[n] << "\n";
  if(logger)
    logger->write(oss.str());
  else
    std::cerr << oss.str();
}

// ============================================================================
// Section: testEvaluate* Helpers and Entry Points
// ============================================================================
// Helper: create a minimal per-thread context for isolated layer testing.
// testEvaluate* helpers

namespace {

  // Construct a VulkanLayerContext from a work context's compiled pipelines + device.
  VulkanLayerContext makeTestLayerCtx(const VulkanWorkContext& ctx) {
    return VulkanLayerContext::fromDevice(
      ctx.dev->device,
      ctx.dev->computeQueue,
      const_cast<mutex&>(ctx.dev->queueMutex),
      ctx.dev->commandPool,
      ctx.dev->info,
      ctx.pipelines->tuneParams,
      ctx.pipelines->pipelineCache);
  }

  // Upload a spatial NCHW buffer with per-channel padding from nnXYLen → paddedSpatialSize.
  // data.size() == batchSize * numChannels * nnXYLen (unpadded)
  VBuf makeTestSpatialInputBuf(
    VulkanWorkContext& ctx,
    const vector<float>& data,
    int batchSize,
    int numChannels,
    int nnXYLen,
    int paddedSpatialSize,
    bool fp16) {
    size_t paddedTotal = static_cast<size_t>(batchSize) * numChannels * paddedSpatialSize;
    auto buf = makeDeviceBuf(ctx.dev->device, ctx.dev->info.memoryProperties, paddedTotal, fp16);
    VkDeviceSize paddedByteSize = paddedTotal * (fp16 ? sizeof(half_t) : sizeof(float));

    auto staging = VulkanBuffer(allocateBuffer(
      ctx.dev->device,
      ctx.dev->info.memoryProperties,
      paddedByteSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));

    void* mapped;
    VK_CHECK(vkMapMemory(ctx.dev->device, staging.memory, 0, paddedByteSize, 0, &mapped));
    if(!fp16) {
      float* dst = static_cast<float*>(mapped);
      std::fill(dst, dst + paddedTotal, 0.0f);
      for(int n = 0; n < batchSize; n++)
        for(int c = 0; c < numChannels; c++)
          for(int xy = 0; xy < nnXYLen; xy++)
            dst[(n * numChannels + c) * paddedSpatialSize + xy] = data[(n * numChannels + c) * nnXYLen + xy];
    } else {
      half_t* dst = static_cast<half_t*>(mapped);
      std::fill(dst, dst + paddedTotal, half_float::half_cast<half_t>(0.0f));
      for(int n = 0; n < batchSize; n++)
        for(int c = 0; c < numChannels; c++)
          for(int xy = 0; xy < nnXYLen; xy++)
            dst[(n * numChannels + c) * paddedSpatialSize + xy] =
              half_float::half_cast<half_t>(data[(n * numChannels + c) * nnXYLen + xy]);
    }
    vkUnmapMemory(ctx.dev->device, staging.memory);
    ctx.submitCopy(staging.buffer, buf->buffer, paddedByteSize);
    return buf;
  }

  // Upload an unpadded [N, XY, C] tensor into padded [N, paddedXY, C] storage.
  VBuf makeTestSpatialNhwcInputBuf(
    VulkanWorkContext& ctx,
    const vector<float>& data,
    int batchSize,
    int numChannels,
    int nnXYLen,
    int paddedSpatialSize,
    bool fp16) {
    size_t paddedTotal = static_cast<size_t>(batchSize) * paddedSpatialSize * numChannels;
    auto buf = makeDeviceBuf(ctx.dev->device, ctx.dev->info.memoryProperties, paddedTotal, fp16);
    VkDeviceSize paddedByteSize = paddedTotal * (fp16 ? sizeof(half_t) : sizeof(float));

    auto staging = VulkanBuffer(allocateBuffer(
      ctx.dev->device,
      ctx.dev->info.memoryProperties,
      paddedByteSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));

    void* mapped;
    VK_CHECK(vkMapMemory(ctx.dev->device, staging.memory, 0, paddedByteSize, 0, &mapped));
    if(!fp16) {
      float* dst = static_cast<float*>(mapped);
      std::fill(dst, dst + paddedTotal, 0.0f);
      for(int n = 0; n < batchSize; n++)
        for(int xy = 0; xy < nnXYLen; xy++)
          for(int c = 0; c < numChannels; c++)
            dst[(n * paddedSpatialSize + xy) * numChannels + c] = data[(n * nnXYLen + xy) * numChannels + c];
    } else {
      half_t* dst = static_cast<half_t*>(mapped);
      std::fill(dst, dst + paddedTotal, half_float::half_cast<half_t>(0.0f));
      for(int n = 0; n < batchSize; n++)
        for(int xy = 0; xy < nnXYLen; xy++)
          for(int c = 0; c < numChannels; c++)
            dst[(n * paddedSpatialSize + xy) * numChannels + c] =
              half_float::half_cast<half_t>(data[(n * nnXYLen + xy) * numChannels + c]);
    }
    vkUnmapMemory(ctx.dev->device, staging.memory);
    ctx.submitCopy(staging.buffer, buf->buffer, paddedByteSize);
    return buf;
  }

  // Allocate a device buffer large enough for padded spatial data.
  VBuf
  makeTestSpatialOutputBuf(VulkanWorkContext& ctx, int batchSize, int numChannels, int paddedSpatialSize, bool fp16) {
    return makeDeviceBuf(
      ctx.dev->device,
      ctx.dev->info.memoryProperties,
      static_cast<size_t>(batchSize) * numChannels * paddedSpatialSize,
      fp16);
  }

  // Exercise the public NCHW input boundary while returning the NHWC storage
  // consumed by every Vulkan layer.
  VBuf makeTestSpatialNchwInputAsNhwcBuf(
    VulkanWorkContext& ctx,
    const vector<float>& data,
    int batchSize,
    int numChannels,
    int nnXYLen,
    int paddedSpatialSize,
    bool fp16) {
    auto nchwBuf = makeTestSpatialInputBuf(ctx, data, batchSize, numChannels, nnXYLen, paddedSpatialSize, fp16);
    auto nhwcBuf = makeTestSpatialOutputBuf(ctx, batchSize, numChannels, paddedSpatialSize, fp16);
    auto kernels =
      VulkanKernels::NchwToNhwc::build(ctx.dev->device, ctx.pipelines->pipelineCache, fp16, ctx.pipelines->tuneParams);
    VulkanKernels::NchwToNhwc::PC pc = {numChannels, numChannels, paddedSpatialSize, batchSize};
    ctx.beginRecording();
    VulkanKernels::NchwToNhwc::dispatch(
      CmdCtx{ctx.cmd, ctx.getPushDescFn()}, kernels, nchwBuf.get(), nhwcBuf.get(), pc);
    ctx.submitAndWait();
    kernels.destroy(ctx.dev->device);
    return nhwcBuf;
  }

  // Download [N, paddedXY, C] storage and remove spatial padding while
  // preserving the caller-visible [N, XY, C] order.
  void downloadTestNhwcBuf(
    VulkanWorkContext& ctx,
    VulkanBuffer* srcBuf,
    vector<float>& dst,
    int batchSize,
    int numChannels,
    int xyLen,
    int paddedLen,
    bool fp16) {
    VkDeviceSize stagingSize = srcBuf->size;
    auto staging = VulkanBuffer(allocateBuffer(
      ctx.dev->device,
      ctx.dev->info.memoryProperties,
      stagingSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    ctx.beginRecording();
    VkBufferCopy r = {0, 0, stagingSize};
    vkCmdCopyBuffer(ctx.cmd, srcBuf->buffer, staging.buffer, 1, &r);
    ctx.submitAndWait();

    void* mapped;
    VK_CHECK(vkMapMemory(ctx.dev->device, staging.memory, 0, stagingSize, 0, &mapped));
    if(!fp16) {
      const float* src = static_cast<const float*>(mapped);
      for(int n = 0; n < batchSize; n++)
        for(int xy = 0; xy < xyLen; xy++)
          for(int c = 0; c < numChannels; c++)
            dst[(n * xyLen + xy) * numChannels + c] = src[(n * paddedLen + xy) * numChannels + c];
    } else {
      const half_t* src = static_cast<const half_t*>(mapped);
      for(int n = 0; n < batchSize; n++)
        for(int xy = 0; xy < xyLen; xy++)
          for(int c = 0; c < numChannels; c++)
            dst[(n * xyLen + xy) * numChannels + c] = (float)src[(n * paddedLen + xy) * numChannels + c];
    }
    vkUnmapMemory(ctx.dev->device, staging.memory);
  }

  // Download native [N, paddedXY, C] storage into the caller-visible
  // unpadded [N, C, XY] order used by NCHW layer tests.
  void downloadTestNhwcAsNchwBuf(
    VulkanWorkContext& ctx,
    VulkanBuffer* srcBuf,
    vector<float>& dst,
    int batchSize,
    int numChannels,
    int xyLen,
    int paddedLen,
    bool fp16) {
    VkDeviceSize stagingSize = srcBuf->size;
    auto staging = VulkanBuffer(allocateBuffer(
      ctx.dev->device,
      ctx.dev->info.memoryProperties,
      stagingSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    ctx.beginRecording();
    VkBufferCopy r = {0, 0, stagingSize};
    vkCmdCopyBuffer(ctx.cmd, srcBuf->buffer, staging.buffer, 1, &r);
    ctx.submitAndWait();

    void* mapped;
    VK_CHECK(vkMapMemory(ctx.dev->device, staging.memory, 0, stagingSize, 0, &mapped));
    if(!fp16) {
      const float* src = static_cast<const float*>(mapped);
      for(int n = 0; n < batchSize; n++)
        for(int c = 0; c < numChannels; c++)
          for(int xy = 0; xy < xyLen; xy++)
            dst[(n * numChannels + c) * xyLen + xy] = src[(n * paddedLen + xy) * numChannels + c];
    } else {
      const half_t* src = static_cast<const half_t*>(mapped);
      for(int n = 0; n < batchSize; n++)
        for(int c = 0; c < numChannels; c++)
          for(int xy = 0; xy < xyLen; xy++)
            dst[(n * numChannels + c) * xyLen + xy] = (float)src[(n * paddedLen + xy) * numChannels + c];
    }
    vkUnmapMemory(ctx.dev->device, staging.memory);
  }

}  // namespace

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {
  VulkanWorkContext ctx;
  ctx.init(0, nnXLen, nnYLen, useFP16 ? enabled_t::True : enabled_t::False, nullptr);
  bool fp16 = ctx.pipelines->usingFP16Storage;
  int nnXYLen = nnXLen * nnYLen;
  int paddedSpatialSize = roundUpToMultipleInt(nnXYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);

  auto layerCtx = makeTestLayerCtx(ctx);
  auto pushFn = ctx.getPushDescFn();

  ConvLayer layer(layerCtx, desc, nnXLen, nnYLen, paddedSpatialSize, fp16);

  size_t inElts = static_cast<size_t>(batchSize) * desc->inChannels * nnXLen * nnYLen;
  size_t outElts = static_cast<size_t>(batchSize) * desc->outChannels * nnXLen * nnYLen;
  if(inputBuffer.size() != inElts)
    throw StringError("testEvaluateConv: unexpected input size");
  outputBuffer.assign(outElts, 0.0f);

  auto inputBuf = useNHWC ? makeTestSpatialNhwcInputBuf(
                              ctx, inputBuffer, batchSize, desc->inChannels, nnXYLen, paddedSpatialSize, fp16)
                          : makeTestSpatialNchwInputAsNhwcBuf(
                              ctx, inputBuffer, batchSize, desc->inChannels, nnXYLen, paddedSpatialSize, fp16);
  auto outputBuf = makeTestSpatialOutputBuf(ctx, batchSize, desc->outChannels, paddedSpatialSize, fp16);

  ScratchBuffers scratch(ctx.dev->device, ctx.dev->info.memoryProperties, fp16, paddedSpatialSize, batchSize);
  size_t elemBytes = fp16 ? sizeof(half_t) : sizeof(float);
  scratch.setPermanentSlots(layer.permanentScratchSlotSizes(batchSize, elemBytes));

  ctx.beginRecording();
  layer.dispatch(CmdCtx{ctx.cmd, pushFn}, &scratch, inputBuf.get(), outputBuf.get(), batchSize);
  ctx.submitAndWait();

  if(useNHWC) {
    downloadTestNhwcBuf(
      ctx, outputBuf.get(), outputBuffer, batchSize, desc->outChannels, nnXYLen, paddedSpatialSize, fp16);
  } else {
    downloadTestNhwcAsNchwBuf(
      ctx, outputBuf.get(), outputBuffer, batchSize, desc->outChannels, nnXYLen, paddedSpatialSize, fp16);
  }
  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  VulkanWorkContext ctx;
  ctx.init(0, nnXLen, nnYLen, useFP16 ? enabled_t::True : enabled_t::False, nullptr);
  bool fp16 = ctx.pipelines->usingFP16Storage;
  int nnXYLen = nnXLen * nnYLen;
  int paddedSpatialSize = roundUpToMultipleInt(nnXYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);

  auto layerCtx = makeTestLayerCtx(ctx);
  auto pushFn = ctx.getPushDescFn();

  size_t elts = static_cast<size_t>(batchSize) * desc->numChannels * nnXLen * nnYLen;
  size_t maskElts = static_cast<size_t>(batchSize) * nnXLen * nnYLen;
  if(inputBuffer.size() != elts)
    throw StringError("testEvaluateBatchNorm: unexpected input size");
  if(maskBuffer.size() != maskElts)
    throw StringError("testEvaluateBatchNorm: unexpected mask size");
  outputBuffer.assign(elts, 0.0f);

  auto inputBuf = useNHWC ? makeTestSpatialNhwcInputBuf(
                              ctx, inputBuffer, batchSize, desc->numChannels, nnXYLen, paddedSpatialSize, fp16)
                          : makeTestSpatialNchwInputAsNhwcBuf(
                              ctx, inputBuffer, batchSize, desc->numChannels, nnXYLen, paddedSpatialSize, fp16);
  auto maskBuf = makeTestSpatialInputBuf(ctx, maskBuffer, batchSize, 1, nnXYLen, paddedSpatialSize, fp16);
  auto outputBuf = makeTestSpatialOutputBuf(ctx, batchSize, desc->numChannels, paddedSpatialSize, fp16);

  if(useNHWC) {
    auto scaleBuf = makeWeightBuf(
      layerCtx.device,
      layerCtx.queue,
      layerCtx.queueMutex,
      layerCtx.commandPool,
      layerCtx.memProps,
      desc->mergedScale,
      fp16);
    auto biasBuf = makeWeightBuf(
      layerCtx.device,
      layerCtx.queue,
      layerCtx.queueMutex,
      layerCtx.commandPool,
      layerCtx.memProps,
      desc->mergedBias,
      fp16);
    // Exercise the optional FP32 [N,C] bias descriptor used by NHWC global
    // pooling fusion. A zero bias preserves this test's expected output while
    // catching specialization-specific descriptor layout/binding mistakes.
    vector<float> dynamicBias(static_cast<size_t>(batchSize) * desc->numChannels, 0.0f);
    auto dynamicBiasBuf = makeWeightBuf(
      layerCtx.device,
      layerCtx.queue,
      layerCtx.queueMutex,
      layerCtx.commandPool,
      layerCtx.memProps,
      dynamicBias,
      false);
    ComputeKernel kernel = VulkanKernels::ScaleBiasMaskActNhwc::build(
      layerCtx.device, layerCtx.pipelineCache, fp16, ACTIVATION_IDENTITY, desc->numChannels % 4 == 0, true);
    VulkanKernels::ScaleBiasMaskActNhwc::PC pc = {desc->numChannels, paddedSpatialSize, batchSize};
    ctx.beginRecording();
    VulkanKernels::ScaleBiasMaskActNhwc::dispatch(
      CmdCtx{ctx.cmd, pushFn},
      kernel,
      inputBuf.get(),
      outputBuf.get(),
      scaleBuf.get(),
      biasBuf.get(),
      maskBuf.get(),
      pc,
      dynamicBiasBuf.get());
    ctx.submitAndWait();
    kernel.destroy(layerCtx.device);
    downloadTestNhwcBuf(
      ctx, outputBuf.get(), outputBuffer, batchSize, desc->numChannels, nnXYLen, paddedSpatialSize, fp16);
  } else {
    ActivationLayerDesc actDesc;
    actDesc.activation = ACTIVATION_IDENTITY;
    BatchNormLayer layer(layerCtx, desc, &actDesc, fp16);
    ScratchBuffers scratch(ctx.dev->device, ctx.dev->info.memoryProperties, fp16, paddedSpatialSize, batchSize);
    ctx.beginRecording();
    layer.dispatch(
      CmdCtx{ctx.cmd, pushFn}, &scratch, inputBuf.get(), outputBuf.get(), maskBuf.get(), paddedSpatialSize, batchSize);
    ctx.submitAndWait();
    downloadTestNhwcAsNchwBuf(
      ctx, outputBuf.get(), outputBuffer, batchSize, desc->numChannels, nnXYLen, paddedSpatialSize, fp16);
  }
  return true;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  if(useNHWC)
    return false;

  VulkanWorkContext ctx;
  ctx.init(0, nnXLen, nnYLen, useFP16 ? enabled_t::True : enabled_t::False, nullptr);
  bool fp16 = ctx.pipelines->usingFP16Storage;
  int nnXYLen = nnXLen * nnYLen;
  int paddedSpatialSize = roundUpToMultipleInt(nnXYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);

  auto layerCtx = makeTestLayerCtx(ctx);
  auto pushFn = ctx.getPushDescFn();

  ResidualBlockVk layer(layerCtx, desc, nnXLen, nnYLen, paddedSpatialSize, fp16);

  int trunkC = desc->preBN.numChannels;
  size_t trunkElts = static_cast<size_t>(batchSize) * trunkC * nnXLen * nnYLen;
  size_t maskElts = static_cast<size_t>(batchSize) * nnXLen * nnYLen;
  if(inputBuffer.size() != trunkElts)
    throw StringError("testEvaluateResidualBlock: unexpected input size");
  if(maskBuffer.size() != maskElts)
    throw StringError("testEvaluateResidualBlock: unexpected mask size");
  outputBuffer = inputBuffer;

  auto trunkBuf =
    makeTestSpatialNchwInputAsNhwcBuf(ctx, outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
  auto scratchBuf = makeTestSpatialOutputBuf(ctx, batchSize, trunkC, paddedSpatialSize, fp16);
  auto maskBuf = makeTestSpatialInputBuf(ctx, maskBuffer, batchSize, 1, nnXYLen, paddedSpatialSize, fp16);

  ScratchBuffers scratch(ctx.dev->device, ctx.dev->info.memoryProperties, fp16, paddedSpatialSize, batchSize);
  size_t elemBytes = fp16 ? sizeof(half_t) : sizeof(float);
  scratch.setPermanentSlots(layer.permanentScratchSlotSizes(batchSize, elemBytes));

  ctx.beginRecording();
  layer.dispatch(
    CmdCtx{ctx.cmd, pushFn}, &scratch, trunkBuf.get(), scratchBuf.get(), maskBuf.get(), nullptr, batchSize);
  ctx.submitAndWait();

  downloadTestNhwcAsNchwBuf(ctx, trunkBuf.get(), outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  if(useNHWC)
    return false;

  VulkanWorkContext ctx;
  ctx.init(0, nnXLen, nnYLen, useFP16 ? enabled_t::True : enabled_t::False, nullptr);
  bool fp16 = ctx.pipelines->usingFP16Storage;
  int nnXYLen = nnXLen * nnYLen;
  int paddedSpatialSize = roundUpToMultipleInt(nnXYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);

  auto layerCtx = makeTestLayerCtx(ctx);
  auto pushFn = ctx.getPushDescFn();

  GlobalPoolingResidualBlockVk layer(layerCtx, desc, nnXLen, nnYLen, paddedSpatialSize, fp16);

  int trunkC = desc->preBN.numChannels;
  size_t trunkElts = static_cast<size_t>(batchSize) * trunkC * nnXLen * nnYLen;
  size_t maskElts = static_cast<size_t>(batchSize) * nnXLen * nnYLen;
  if(inputBuffer.size() != trunkElts)
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input size");
  if(maskBuffer.size() != maskElts)
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask size");
  outputBuffer = inputBuffer;

  auto trunkBuf =
    makeTestSpatialNchwInputAsNhwcBuf(ctx, outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
  auto scratchBuf = makeTestSpatialOutputBuf(ctx, batchSize, trunkC, paddedSpatialSize, fp16);
  auto maskBuf = makeTestSpatialInputBuf(ctx, maskBuffer, batchSize, 1, nnXYLen, paddedSpatialSize, fp16);

  // Compute maskSum on CPU
  vector<float> maskSumData(batchSize);
  for(int n = 0; n < batchSize; n++) {
    float s = 0.0f;
    for(int xy = 0; xy < nnXYLen; xy++)
      s += maskBuffer[n * nnXYLen + xy];
    maskSumData[n] = s;
  }
  auto maskSumBuf = ctx.makeInputBuf(maskSumData);

  ScratchBuffers scratch(ctx.dev->device, ctx.dev->info.memoryProperties, fp16, paddedSpatialSize, batchSize);
  size_t elemBytes = fp16 ? sizeof(half_t) : sizeof(float);
  scratch.setPermanentSlots(layer.permanentScratchSlotSizes(batchSize, elemBytes));

  ctx.beginRecording();
  layer.dispatch(
    CmdCtx{ctx.cmd, pushFn}, &scratch, trunkBuf.get(), scratchBuf.get(), maskBuf.get(), maskSumBuf.get(), batchSize);
  ctx.submitAndWait();

  downloadTestNhwcAsNchwBuf(ctx, trunkBuf.get(), outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
  return true;
}

#endif  // USE_VULKAN_BACKEND

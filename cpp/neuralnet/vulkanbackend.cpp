#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkanbackend.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <ghc/filesystem.hpp>
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
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanincludes.h"
#include "../neuralnet/vulkankernels.h"
#include "../neuralnet/vulkantuner.h"
#ifdef OS_IS_UNIX_OR_APPLE
#include <unistd.h>
#endif
#ifdef OS_IS_WINDOWS
#include <windows.h>
#include <codecvt>
#endif

using namespace VulkanHelpers;
using half_t = half_float::half;
using std::lock_guard;
using std::mutex;
using std::to_string;

// ============================================================================
// FP16 CONVENTIONS (same as OpenCL backend)
//
// IMPORTANT: "useFP16" / "usingFP16Storage" / the USE_FP16_STORAGE shader spec
// constant all mean FP16 *storage* by default. Tensors are stored as 16-bit
// floats and most shaders widen loads to FP32, do arithmetic in FP32, and narrow
// stores back to FP16.
//
// A few explicitly named experimental kernels can instead use full FP16 compute
// (including FP16 accumulation) when shaderFloat16 is enabled and the tuner
// selects them. Callers should still treat useFP16 as the storage-mode contract.
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
// NHWC throughout; external channel-first inputs convert once on-device
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

void NeuralNet::printDevices() {
  VkInstance inst = getOrCreateInstance();
  VulkanHelpers::printDevices(inst, nullptr);
}

namespace {

  std::string_view getenvStringView(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string_view() : std::string_view(value, std::strlen(value));
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

  // Returns true iff `path` exists and has an mtime strictly older than the
  // current executable's mtime. Used for pipeline-cache and tune-cache
  // invalidation: a rebuilt katago may ship different SPIR-V or different tuner
  // logic, so persisted state from before the rebuild is potentially stale and
  // must not be reused. Returns false on any error (unknown OS, missing files,
  // unreadable exe path) — the caller then keeps whatever data it already had.
  bool fileIsOlderThanExecutable(const string& path) {
    namespace gfs = ghc::filesystem;
    std::error_code ec;
#if defined(OS_IS_UNIX_OR_APPLE)
    constexpr int bufSize = 2048;
    char exeBuf[bufSize];
    ssize_t exeLen = readlink("/proc/self/exe", exeBuf, bufSize - 1);
    if(exeLen <= 0)
      return false;
    auto exeMtime = gfs::last_write_time(gfs::u8path(string(exeBuf, exeLen)), ec);
    if(ec)
      return false;
    auto fileMtime = gfs::last_write_time(gfs::u8path(path), ec);
    if(ec)
      return false;
    return exeMtime > fileMtime;
#elif defined(OS_IS_WINDOWS)
    constexpr size_t bufSize = MAX_PATH + 2048;
    wchar_t exeBuf[bufSize];
    DWORD exeLen = GetModuleFileNameW(NULL, exeBuf, (DWORD)bufSize);
    if(exeLen == 0 || exeLen >= bufSize)
      return false;
    std::wstring_convert<std::codecvt_utf8<wchar_t> > conv;
    string exePath = conv.to_bytes(wstring(exeBuf, exeLen));
    auto exeMtime = gfs::last_write_time(gfs::u8path(exePath), ec);
    if(ec)
      return false;
    auto fileMtime = gfs::last_write_time(gfs::u8path(path), ec);
    if(ec)
      return false;
    return exeMtime > fileMtime;
#else
    (void)path;
    return false;
#endif
  }

  // Reconcile coopmat tune params against the runtime device. A tune file may
  // predate this GPU (explicit --vulkan-tuner-file, or a cache reused across
  // devices with the same tuning-key but different coopmat capabilities), or be
  // hand-edited. resolve*Variant only checks supportsCoopmat1F16 + the enable
  // flag, so an enabled config with a TM/TN/TK the device does not report, or a
  // WARP != this GPU's subgroup size, would build a broken/mis-mapped pipeline.
  // Disable coopmat and request a retune in those cases. Returns true if
  // anything was disabled.
  bool reconcileCoopmatParams(VulkanTuneParams& cfg, const VulkanDeviceInfo& info, Logger* logger) {
    if(
      cfg.enableWinogradGemmCoopmat1 == 0 && cfg.enableWinogradGemmCoopmat2 == 0 &&
      cfg.enableWinogradGemmCoopmat1AccF16 == 0 && cfg.enableWinogradGemmCoopmat2AccF16 == 0 &&
      cfg.nhwcConv3x3UseCoopmat2 == 0 && cfg.nhwcConv5x5UseCoopmat2 == 0)
      return false;

    auto shapeSupported = [&](const vector<CoopmatShape>& shapes, int32_t tm, int32_t tn, int32_t tk) {
      for(const auto& s: shapes)
        if(s.m == (uint32_t)tm && s.n == (uint32_t)tn && s.k == (uint32_t)tk)
          return true;
      return false;
    };
    auto disable = [&](std::string_view which, const string& why) {
      if(logger != nullptr)
        logger->write(
          string("Vulkan tune file enables ") + string(which) + " but " + why + "; falling back and re-tuning.");
    };
    auto coopmat2TileSupported =
      [&](
        bool supports, const vector<Coopmat2FlexShape>& shapes, int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
        if(!supports)
          return false;
        if(info.coopmat2ReservedSharedBytes > info.properties.limits.maxComputeSharedMemorySize)
          return false;
        if(info.coopmat2MaxWorkgroupSize > 0 && (uint32_t)blockSize > info.coopmat2MaxWorkgroupSize)
          return false;
        if(info.coopmat2MaxFlexDimension > 0) {
          uint32_t maxDim = std::max((uint32_t)bm, std::max((uint32_t)bn, (uint32_t)bk));
          if(maxDim > info.coopmat2MaxFlexDimension)
            return false;
        }
        for(const auto& s: shapes) {
          if(
            blockSize == (int32_t)s.workgroupInvocations && bm % (int32_t)s.mGranularity == 0 &&
            bn % (int32_t)s.nGranularity == 0 && bk % (int32_t)s.kGranularity == 0)
            return true;
        }
        return false;
      };

    bool disabledAny = false;
    if(cfg.enableWinogradGemmCoopmat1 != 0) {
      string why;
      if(!info.supportsCoopmat1F16)
        why = "this GPU does not support cooperative matrices";
      else if(cfg.coopmat1Warp != (int32_t)info.subgroupSize)
        why = "coopmat1Warp (" + Global::intToString(cfg.coopmat1Warp) + ") does not match runtime subgroup size (" +
              Global::intToString((int)info.subgroupSize) + ")";
      else if(!shapeSupported(info.coopmatShapes, cfg.coopmat1TM, cfg.coopmat1TN, cfg.coopmat1TK))
        why = "its coopmat fragment shape is not one this GPU reports";
      if(!why.empty()) {
        disable("winogradGemmCoopmat1", why);
        cfg.enableWinogradGemmCoopmat1 = 0;
        disabledAny = true;
      }
    }
    if(cfg.enableWinogradGemmCoopmat2 != 0) {
      string why;
      if(!info.supportsCoopmat2F16)
        why = "this GPU does not support cooperative matrix2";
      else if(!coopmat2TileSupported(
                info.supportsCoopmat2F16,
                info.coopmat2FlexShapes,
                cfg.coopmat2BlockSize,
                cfg.coopmat2BM,
                cfg.coopmat2BN,
                cfg.coopmat2BK))
        why = "its coopmat2 flexible dimensions are not one this GPU reports";
      if(!why.empty()) {
        disable("winogradGemmCoopmat2", why);
        cfg.enableWinogradGemmCoopmat2 = 0;
        disabledAny = true;
      }
    }
    if(cfg.enableWinogradGemmCoopmat1AccF16 != 0) {
      string why;
      if(!info.supportsCoopmat1F16AccF16)
        why = "this GPU does not support cooperative matrices with FP16 accumulation";
      else if(cfg.coopmat1AccF16Warp != (int32_t)info.subgroupSize)
        why = "coopmat1AccF16Warp (" + Global::intToString(cfg.coopmat1AccF16Warp) +
              ") does not match runtime subgroup size (" + Global::intToString((int)info.subgroupSize) + ")";
      else if(!shapeSupported(info.coopmatAccF16Shapes, cfg.coopmat1AccF16TM, cfg.coopmat1AccF16TN, cfg.coopmat1AccF16TK))
        why = "its coopmatAccF16 fragment shape is not one this GPU reports";
      if(!why.empty()) {
        disable("winogradGemmCoopmat1AccF16", why);
        cfg.enableWinogradGemmCoopmat1AccF16 = 0;
        disabledAny = true;
      }
    }
    if(cfg.enableWinogradGemmCoopmat2AccF16 != 0) {
      string why;
      if(!info.supportsCoopmat2F16AccF16)
        why = "this GPU does not support cooperative matrix2 with FP16 accumulation";
      else if(!coopmat2TileSupported(
                info.supportsCoopmat2F16AccF16,
                info.coopmat2AccF16FlexShapes,
                cfg.coopmat2AccF16BlockSize,
                cfg.coopmat2AccF16BM,
                cfg.coopmat2AccF16BN,
                cfg.coopmat2AccF16BK))
        why = "its coopmat2AccF16 flexible dimensions are not one this GPU reports";
      if(!why.empty()) {
        disable("winogradGemmCoopmat2AccF16", why);
        cfg.enableWinogradGemmCoopmat2AccF16 = 0;
        disabledAny = true;
      }
    }
    if(cfg.nhwcConv3x3UseCoopmat2 != 0) {
      string why;
      if(!info.supportsCoopmat2F16)
        why = "this GPU does not support cooperative matrix2 with FP32 accumulation";
      else if(!coopmat2TileSupported(
                info.supportsCoopmat2F16,
                info.coopmat2FlexShapes,
                cfg.conv3x3NhwcCoopmat2BlockSize,
                cfg.conv3x3NhwcCoopmat2BM,
                cfg.conv3x3NhwcCoopmat2BN,
                cfg.conv3x3NhwcCoopmat2BK))
        why = "its coopmat2 flexible dimensions are not one this GPU reports";
      else if(
        (size_t)info.coopmat2ReservedSharedBytes +
          VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::sharedBytes(
            cfg.conv3x3NhwcCoopmat2BM, cfg.conv3x3NhwcCoopmat2BK) >
        info.properties.limits.maxComputeSharedMemorySize)
        why = "its coopmat2 shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
      if(!why.empty()) {
        // Fall back to the coopmat1 conv rather than disabling the 3x3 path. The
        // workgroup-scope fragment is the whole tile here, so an unreported shape
        // would produce wrong results, not merely slow ones.
        // A GEMM_VARIANTS filter can temporarily hide the FP32-accumulating
        // coopmat2 convolution. Keep its measured tile valid so an FP16-only
        // run does not turn a later unrestricted run into a cache miss.
        if(info.filterDisabledCoopmat2F16 && !info.supportsCoopmat2F16) {
          // This is only an in-memory fallback. Preserve the cached selector
          // too: the runtime support check already prevents this shader from
          // being built, and a filtered run must not overwrite the default
          // selector when it later saves a Winograd-transform re-tune.
        } else {
          disable("nhwcConv3x3UseCoopmat2", why);
          cfg.nhwcConv3x3UseCoopmat2 = 0;
          cfg.conv3x3NhwcCoopmat2TunerValueValid = 0;
          disabledAny = true;
        }
      }
    }
    if(cfg.nhwcConv5x5UseCoopmat2 != 0) {
      string why;
      if(!info.supportsCoopmat2F16)
        why = "this GPU does not support cooperative matrix2 with FP32 accumulation";
      else if(!coopmat2TileSupported(
                info.supportsCoopmat2F16,
                info.coopmat2FlexShapes,
                cfg.conv5x5NhwcCoopmat2BlockSize,
                cfg.conv5x5NhwcCoopmat2BM,
                cfg.conv5x5NhwcCoopmat2BN,
                cfg.conv5x5NhwcCoopmat2BK))
        why = "its coopmat2 flexible dimensions are not one this GPU reports";
      else if(
        (size_t)info.coopmat2ReservedSharedBytes +
          VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::sharedBytes(
            cfg.conv5x5NhwcCoopmat2BM, cfg.conv5x5NhwcCoopmat2BK) >
        info.properties.limits.maxComputeSharedMemorySize)
        why = "its coopmat2 shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
      if(!why.empty()) {
        if(info.filterDisabledCoopmat2F16 && !info.supportsCoopmat2F16) {
          // See the corresponding 3x3 case above.
        } else {
          disable("nhwcConv5x5UseCoopmat2", why);
          cfg.nhwcConv5x5UseCoopmat2 = 0;
          cfg.conv5x5NhwcCoopmat2TunerValueValid = 0;
          disabledAny = true;
        }
      }
    }
    return disabledAny;
  }

  bool vulkanBatchProfileEnabled() {
    static const bool cached = []() {
      std::string_view v = getenvStringView("KATAGO_VULKAN_PROFILE_BATCHES");
      return !v.empty() && v[0] != '0';
    }();
    return cached;
  }

}  // namespace

struct CompiledPipelines {
  VkDevice device;
  bool usingFP16Storage;
  bool usingFP16Compute;
  VulkanTuneParams tuneParams;
  VkPipelineCache pipelineCache;
  string pipelineCacheFile;
  Logger* logger;

  static int actToIdx(int activation) {
    switch(activation) {
      case ACTIVATION_IDENTITY:
        return 0;
      case ACTIVATION_RELU:
        return 1;
      case ACTIVATION_MISH:
        return 2;
      case ACTIVATION_SILU:
        return 3;
      case ACTIVATION_MISH_SCALE8:
        return 4;
      default:
        throw StringError("Unknown activation " + std::to_string(activation));
    }
  }

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

        // Discard cached data if the katago executable is newer than the cache file.
        // A rebuilt executable may have different SPIR-V embedded, so the old cache
        // is potentially incompatible and must not be used.
        if(!initialData.empty() && fileIsOlderThanExecutable(pipelineCacheFile)) {
          if(logger != nullptr)
            logger->write("Vulkan: pipeline cache is older than the executable, ignoring: " + pipelineCacheFile);
          initialData.clear();
        }
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

  uint32_t fp16s() const { return usingFP16Storage ? 1u : 0u; }

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
  std::vector<int> gpuIdxsNeedingGemmVariantsTune;
  std::vector<int> gpuIdxsNeedingWinogradTransformsTune;

  // Per-(model,vulkanTunerFile,vulkanReTunePerBoardSize) state remembered so a
  // later tuneSelf() can save to the same default path. Only populated when the
  // ctor was given a non-null modelDescForTuneLoad.
  bool autoTuneEnabled = false;
  bool autoTuneReTunePerBoardSize = false;
  std::string autoTuneExplicitFile;  // empty if no explicit override

  ComputeContext(
    VkInstance inst,
    const vector<int>& gpuIdxs,
    Logger* logger_,
    int nnX,
    int nnY,
    const string& homeDataDirOverride_,
    enabled_t fp16Mode,
    bool forceDisableFP16 = false,
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
      autoTuneReTunePerBoardSize(vulkanReTunePerBoardSize),
      autoTuneExplicitFile(vulkanTunerFile) {
    auto allDeviceInfos = VulkanDeviceInfo::getAllDeviceInfosOnSystem(inst, logger);

    bool requestFP16Storage = (fp16Mode == enabled_t::True || fp16Mode == enabled_t::Auto) && !forceDisableFP16;
    bool requestFP16Compute = requestFP16Storage;

    devicesContext =
      new VulkanDevicesContext(inst, allDeviceInfos, gpuIdxs, requestFP16Storage, requestFP16Compute, logger);

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
      auto requestWinogradTransformsRetuneForDevice = [&]() {
        if(modelDescForTuneLoad == nullptr)
          return;
        if(
          std::find(
            gpuIdxsNeedingWinogradTransformsTune.begin(),
            gpuIdxsNeedingWinogradTransformsTune.end(),
            dev->info.gpuIdx) == gpuIdxsNeedingWinogradTransformsTune.end())
          gpuIdxsNeedingWinogradTransformsTune.push_back(dev->info.gpuIdx);
      };
      auto maybeRequestWinogradTransformsRetune = [&](const VulkanTuneParams& loaded) {
        if(modelDescForTuneLoad == nullptr)
          return;
        const int32_t requiredMask = VulkanTuner::requiredKernelMask(
          modelDescForTuneLoad, dev->info, fp16Storage, fp16Compute, true, false);
        if((requiredMask & VulkanTuner::TUNED_NHWC_WINOGRAD) == 0 ||
           VulkanTuner::nhwcWinogradTransformsTunedForCurrentLayout(loaded, fp16Storage))
          return;
        if(logger != nullptr)
          logger->write(
            "Vulkan tune file NHWC Winograd transforms were tuned for a different Winograd GEMM layout; "
            "re-tuning only winogradTransformNhwc and winogradUntransformNhwc.");
        requestWinogradTransformsRetuneForDevice();
      };

      if(!vulkanTunerFile.empty()) {
        auto requestRetuneForDevice = [&]() {
          if(modelDescForTuneLoad == nullptr)
            return;
          if(
            std::find(gpuIdxsNeedingTune.begin(), gpuIdxsNeedingTune.end(), dev->info.gpuIdx) ==
            gpuIdxsNeedingTune.end())
            gpuIdxsNeedingTune.push_back(dev->info.gpuIdx);
        };
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
          if(!loaded.isValid())
            handleBadExplicitTuneFile("loaded vulkan tune params were not valid");
          else {
            if(loaded.normalizeGemmVariantSelectionsForDevice(dev->info, fp16Storage, fp16Compute) && logger != nullptr)
              logger->write(
                "Vulkan tune file contained a GEMM selection unavailable on the current hardware/filter; "
                "using the tiered fallback: " + vulkanTunerFile);
            loadedExplicitFile = true;
          }
        } catch(const IOError& e) {
          handleBadExplicitTuneFile(e.what());
        }

        if(loadedExplicitFile) {
          bool requestedFullRetune = false;
          if(reconcileCoopmatParams(loaded, dev->info, logger)) {
            requestRetuneForDevice();
            requestedFullRetune = true;
          }
          const int32_t requiredMask = VulkanTuner::requiredKernelMask(
            modelDescForTuneLoad,
            dev->info,
            fp16Storage,
            fp16Compute,
            vulkanUseNhwc(),
            false);
          const bool missingNhwcWinogradTune =
            vulkanUseNhwc() && (requiredMask & VulkanTuner::TUNED_NHWC_WINOGRAD) != 0 &&
            (loaded.conv3x3NhwcWinogradTunerValueValid == 0 || loaded.conv5x5NhwcWinogradTunerValueValid == 0);
          const bool incompleteGemmVariants = !VulkanTuner::gemmVariantTuningCompleteForCurrentHardware(
            loaded, dev->info, fp16Storage, fp16Compute, requiredMask, vulkanUseNhwc());
          if(
            !requestedFullRetune &&
            (((loaded.tunedKernelMask & requiredMask) != requiredMask) ||
             missingNhwcWinogradTune ||
             incompleteGemmVariants))
            requestRetuneForDevice();
          if(!requestedFullRetune)
            maybeRequestWinogradTransformsRetune(loaded);
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
          const int32_t requiredMask = VulkanTuner::requiredKernelMask(
            modelDescForTuneLoad,
            dev->info,
            fp16Storage,
            fp16Compute,
            vulkanUseNhwc(),
            false);
          const bool missingNhwcWinogradTune =
            vulkanUseNhwc() && (requiredMask & VulkanTuner::TUNED_NHWC_WINOGRAD) != 0 &&
            (perDeviceParams.conv3x3NhwcWinogradTunerValueValid == 0 ||
             perDeviceParams.conv5x5NhwcWinogradTunerValueValid == 0);
          const bool incompleteGemmVariants = !VulkanTuner::gemmVariantTuningCompleteForCurrentHardware(
            perDeviceParams, dev->info, fp16Storage, fp16Compute, requiredMask, vulkanUseNhwc());
          if(
            (perDeviceParams.tunedKernelMask & requiredMask) != requiredMask ||
            missingNhwcWinogradTune ||
            incompleteGemmVariants)
            gpuIdxsNeedingTune.push_back(dev->info.gpuIdx);
          maybeRequestWinogradTransformsRetune(perDeviceParams);
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
    bool fp16,
    bool fp16Compute,
    VulkanTuneParams& out) const {
    auto attempt = [&](int xLen, int yLen) -> bool {
      string dir = VulkanTuner::defaultDirectory(true, homeDataDirOverride);
      string path = dir + "/" + VulkanTuner::defaultFileName(deviceName, tuneKey, xLen, yLen, modelDesc, fp16);
      try {
        // Reject tune files older than the current executable. A rebuild may
        // ship new kernel shapes or new default tunables, and silently reusing
        // pre-rebuild parameters can leave us tuned for kernels that no longer
        // exist (or miss tunables that do).
        if(fileIsOlderThanExecutable(path)) {
          if(logger != nullptr)
            logger->write("Vulkan tune file is older than the executable, ignoring (re-tuning): " + path);
          return false;
        }
        VulkanTuneParams loaded = VulkanTuneParams::load(path);
        if(!loaded.isValid()) {
          if(logger != nullptr)
            logger->write("Vulkan tune file exists but failed isValid(), ignoring: " + path);
          return false;
        }
        if(loaded.normalizeGemmVariantSelectionsForDevice(deviceInfo, fp16, fp16Compute) && logger != nullptr)
          logger->write(
            "Vulkan tune file contained a GEMM selection unavailable on the current hardware/filter; "
            "using the tiered fallback: " + path);
        // Coopmat: on any device-incompatibility, treat as a cache miss so the
        // caller re-tunes for this GPU rather than running with disabled coopmat.
        if(reconcileCoopmatParams(loaded, deviceInfo, logger))
          return false;
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
  bool needsGemmVariantsTuning() const { return !gpuIdxsNeedingGemmVariantsTune.empty(); }
  bool needsWinogradTransformsTuning() const { return !gpuIdxsNeedingWinogradTransformsTune.empty(); }

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

  void tuneWinogradTransformsSelf(const ModelDesc* modelDesc, int batchSize, int benchIters, bool verboseTuner);
  void tuneGemmVariantsSelf(const ModelDesc* modelDesc, int batchSize, int benchIters, bool verboseTuner);

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
    /*forceDisableFP16=*/false,
    &(loadedModel->modelDesc),
    vulkanTunerFile,
    vulkanReTunePerBoardSize);

  if(ctx->needsTuning() || ctx->needsGemmVariantsTuning() || ctx->needsWinogradTransformsTuning()) {
    int tuneBatchSize = VulkanTuner::DEFAULT_BATCH_SIZE;
    if(cfg.contains("numSearchThreads")) {
      // Search batches typically range from 1 up to about numSearchThreads,
      // so tune at the upper end of the expected runtime range.
      tuneBatchSize = std::max(1, cfg.getInt("numSearchThreads", 1, 65536));
    }
    if(ctx->needsTuning()) {
      ctx->tuneSelf(
        &(loadedModel->modelDesc),
        tuneBatchSize,
        VulkanTuner::DEFAULT_WINOGRAD_3X3_TILE_SIZE,
        /*benchIters=*/500,
        /*verboseTuner=*/false,
        /*full=*/false,
        /*saveFileOverride=*/"");
    }
    if(ctx->needsGemmVariantsTuning())
      ctx->tuneGemmVariantsSelf(&(loadedModel->modelDesc), tuneBatchSize, /*benchIters=*/500, /*verboseTuner=*/false);
    if(ctx->needsWinogradTransformsTuning())
      ctx->tuneWinogradTransformsSelf(
        &(loadedModel->modelDesc), tuneBatchSize, /*benchIters=*/500, /*verboseTuner=*/false);
  }

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* ctx) {
  delete ctx;
}

ComputeContext* NeuralNet::createComputeContextForVulkanTuner(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode) {
  if(gpuIdxs.empty())
    throw StringError("NeuralNet::createComputeContextForVulkanTuner - no GPUs specified");
  VkInstance inst = getOrCreateInstance();
  return new ComputeContext(
    inst,
    gpuIdxs,
    logger,
    nnXLen,
    nnYLen,
    homeDataDirOverride,
    useFP16Mode,
    /*forceDisableFP16=*/false,
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

void VulkanWorkContext::warmupDispatches(double targetSeconds, int maxIters, const std::function<void()>& recordOne) {
  if(targetSeconds <= 0.0 || maxIters <= 0)
    return;

  // Split into small submissions to avoid GPU watchdog timeouts (TDR).
  // This is intentionally untimed: it settles cache/pipeline state and gives
  // the GPU clock governor sustained work before the measured probe starts.
  static constexpr int CHUNK_SIZE = 64;
  auto t0 = std::chrono::steady_clock::now();
  int remaining = maxIters;
  while(remaining > 0) {
    int chunk = std::min(remaining, CHUNK_SIZE);
    remaining -= chunk;

    beginRecording();
    for(int i = 0; i < chunk; i++)
      recordOne();
    submitAndWait();

    auto t1 = std::chrono::steady_clock::now();
    if(std::chrono::duration<double>(t1 - t0).count() >= targetSeconds)
      break;
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
      string dir = VulkanTuner::defaultDirectory(true, homeDataDirOverride);
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
        if(fileIsOlderThanExecutable(outputPath)) {
          if(logger != nullptr)
            logger->write("Vulkan tune file is older than the executable, ignoring (re-tuning): " + outputPath);
        } else {
          try {
            VulkanTuneParams loaded = VulkanTuneParams::load(outputPath);
            if(!loaded.isValid()) {
              if(logger != nullptr)
                logger->write("Vulkan tune file exists but failed isValid(), ignoring: " + outputPath);
            } else {
              if(
                loaded.normalizeGemmVariantSelectionsForDevice(
                  dev->info, pipelines->usingFP16Storage, pipelines->usingFP16Compute) &&
                logger != nullptr)
                logger->write(
                  "Vulkan tune file contained a GEMM selection unavailable on the current hardware/filter; "
                  "using the tiered fallback: " + outputPath);
              if(!reconcileCoopmatParams(loaded, dev->info, logger)) {
                pipelines->tuneParams = loaded;
                if(logger != nullptr)
                  logger->write("Loaded existing vulkan tuning parameters for incremental tuning from: " + outputPath);
              }
            }
          } catch(const IOError& e) {
            if(logger != nullptr)
              logger->write(
                "Vulkan tune file could not be loaded, ignoring and re-tuning: " + outputPath + " (" + e.what() +
                ")");
          }
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
          "Vulkan Winograd output tile changed from " +
          Global::intToString(initialParams.nhwcWinograd3x3OutTile) + " to " + Global::intToString(winograd3x3OutTile) +
          "; invalidating Winograd tuning.");
      initialParams.tunedKernelMask &= ~VulkanTuner::TUNED_NHWC_WINOGRAD;
      initialParams.nhwcWinogradTransformTunedLayout.clear();
      initialParams.winogradGemmDot2TunerValueValid = 0;
      initialParams.winogradGemmDot2AccF16TunerValueValid = 0;
      initialParams.winogradGemmCoopmat1TunerValueValid = 0;
      initialParams.winogradGemmCoopmat1AccF16TunerValueValid = 0;
      initialParams.winogradGemmCoopmat2TunerValueValid = 0;
      initialParams.winogradGemmCoopmat2AccF16TunerValueValid = 0;
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

  for(int gpuIdx: toTune) {
    gpuIdxsNeedingGemmVariantsTune.erase(
      std::remove(gpuIdxsNeedingGemmVariantsTune.begin(), gpuIdxsNeedingGemmVariantsTune.end(), gpuIdx),
      gpuIdxsNeedingGemmVariantsTune.end());
    gpuIdxsNeedingWinogradTransformsTune.erase(
      std::remove(gpuIdxsNeedingWinogradTransformsTune.begin(), gpuIdxsNeedingWinogradTransformsTune.end(), gpuIdx),
      gpuIdxsNeedingWinogradTransformsTune.end());
  }
  gpuIdxsNeedingTune.clear();
}

void ComputeContext::tuneGemmVariantsSelf(
  const ModelDesc* modelDesc,
  int batchSize,
  int benchIters,
  bool verboseTuner) {
  if(handlesAlreadyCreated)
    throw StringError(
      "ComputeContext::tuneGemmVariantsSelf called after ComputeHandle creation - pipelines already compiled with "
      "stale spec constants");
  testAssert(modelDesc != nullptr);

  std::vector<int> toTune = gpuIdxsNeedingGemmVariantsTune;
  std::map<std::string, VulkanTuneParams> tunedParamsByKey;
  auto needsWinogradTransformsRetune = [&](const InitializedVulkanDevice* dev, const CompiledPipelines* pipelines) {
    const int32_t requiredMask = VulkanTuner::requiredKernelMask(
      modelDesc, dev->info, pipelines->usingFP16Storage, pipelines->usingFP16Compute, true, false);
    return (requiredMask & VulkanTuner::TUNED_NHWC_WINOGRAD) != 0 &&
      !VulkanTuner::nhwcWinogradTransformsTunedForCurrentLayout(
        pipelines->tuneParams, pipelines->usingFP16Storage);
  };

  for(int gpuIdx: toTune) {
    const InitializedVulkanDevice* dev = devicesContext->findGpuExn(gpuIdx);
    const std::string gpuName(dev->info.properties.deviceName);
    const std::string tuneKey = tuningDeviceKey(dev);

    auto alreadyTuned = tunedParamsByKey.find(tuneKey);
    if(alreadyTuned != tunedParamsByKey.end()) {
      CompiledPipelines* pipelines = compiledPipelinesByDevice.at(dev->info.physicalDevice);
      pipelines->tuneParams = alreadyTuned->second;
      if(
        needsWinogradTransformsRetune(dev, pipelines) &&
        std::find(gpuIdxsNeedingWinogradTransformsTune.begin(), gpuIdxsNeedingWinogradTransformsTune.end(), gpuIdx) ==
          gpuIdxsNeedingWinogradTransformsTune.end())
        gpuIdxsNeedingWinogradTransformsTune.push_back(gpuIdx);
      if(logger != nullptr)
        logger->write(
          "Reusing Vulkan GEMM variant update for device " + Global::intToString(gpuIdx) +
          " (same tuning key as an earlier updated GPU): " + gpuName);
      continue;
    }

    if(logger != nullptr) {
      logger->write(
        "Updating Vulkan GEMM variant candidates for " + gpuName + " for batch " + Global::intToString(batchSize));
    }

    VulkanTuner::TuningContext session;
    session.logger = logger;
    session.initBorrowed(this, gpuIdx, logger);
    session.fp16Storage = session.resolvedFP16Storage();
    session.fp16Compute = session.resolvedFP16Compute();

    CompiledPipelines* pipelines = compiledPipelinesByDevice.at(dev->info.physicalDevice);
    VulkanTuneParams results = pipelines->tuneParams;
    bool changed = VulkanTuner::tuneGemmVariantsForCurrentHardware(
      session, modelDesc, batchSize, nnXLen, nnYLen, benchIters, std::cerr, verboseTuner, results);
    session.destroy();
    tunedParamsByKey[tuneKey] = results;

    if(changed) {
      pipelines->tuneParams = results;
      if(
        needsWinogradTransformsRetune(dev, pipelines) &&
        std::find(gpuIdxsNeedingWinogradTransformsTune.begin(), gpuIdxsNeedingWinogradTransformsTune.end(), gpuIdx) ==
          gpuIdxsNeedingWinogradTransformsTune.end())
        gpuIdxsNeedingWinogradTransformsTune.push_back(gpuIdx);

      string outputPath;
      if(!autoTuneExplicitFile.empty()) {
        outputPath = autoTuneExplicitFile;
      } else {
        string dir = VulkanTuner::defaultDirectory(true, homeDataDirOverride);
        outputPath =
          dir + "/" +
          VulkanTuner::defaultFileName(gpuName, tuneKey, nnXLen, nnYLen, modelDesc, pipelines->usingFP16Storage);
      }
      VulkanTuneParams::save(outputPath, results);
      if(logger != nullptr) {
        std::ostringstream s;
        VulkanTuner::appendTuneParamsSummary(s, results, modelDesc);
        logger->write(
          "Using vulkan tuning parameters for GPU " + Global::intToString(gpuIdx) + " (" + gpuName + "): " + s.str());
        logger->write("Done updating Vulkan GEMM variants, saved results to " + outputPath);
      }
    } else if(logger != nullptr) {
      logger->write("Vulkan GEMM variant update did not change the loaded tuning parameters.");
    }
  }

  gpuIdxsNeedingGemmVariantsTune.clear();
}

void ComputeContext::tuneWinogradTransformsSelf(
  const ModelDesc* modelDesc,
  int batchSize,
  int benchIters,
  bool verboseTuner) {
  if(handlesAlreadyCreated)
    throw StringError(
      "ComputeContext::tuneWinogradTransformsSelf called after ComputeHandle creation - pipelines already compiled "
      "with "
      "stale spec constants");
  testAssert(modelDesc != nullptr);

  std::vector<int> toTune = gpuIdxsNeedingWinogradTransformsTune;
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
          "Reusing Winograd transform re-tune for device " + Global::intToString(gpuIdx) +
          " (same tuning key as an earlier tuned GPU): " + gpuName);
      continue;
    }

    CompiledPipelines* pipelines = compiledPipelinesByDevice.at(dev->info.physicalDevice);
    const bool transformsAlreadyTuned =
      VulkanTuner::nhwcWinogradTransformsTunedForCurrentLayout(pipelines->tuneParams, pipelines->usingFP16Storage);
    if(transformsAlreadyTuned) {
      tunedParamsByKey[tuneKey] = pipelines->tuneParams;
      continue;
    }

    if(logger != nullptr) {
      logger->write(
        "Re-tuning Vulkan NHWC Winograd transform/untransform for selected Winograd GEMM layout on " + gpuName + " for batch " +
        Global::intToString(batchSize));
    }

    VulkanTuner::TuningContext session;
    session.logger = logger;
    session.initBorrowed(this, gpuIdx, logger);
    session.fp16Storage = session.resolvedFP16Storage();
    session.fp16Compute = session.resolvedFP16Compute();

    VulkanTuneParams results = pipelines->tuneParams;
    bool retuned = VulkanTuner::retuneNhwcWinogradTransformsForCurrentLayout(
      session, modelDesc, batchSize, nnXLen, nnYLen, benchIters, std::cerr, verboseTuner, results);
    session.destroy();

    if(retuned) {
      pipelines->tuneParams = results;
      tunedParamsByKey[tuneKey] = results;
      if(!autoTuneExplicitFile.empty()) {
        VulkanTuneParams::save(autoTuneExplicitFile, results);
        if(logger != nullptr)
          logger->write("Done re-tuning Vulkan Winograd transforms, saved results to " + autoTuneExplicitFile);
      } else {
        string dir = VulkanTuner::defaultDirectory(true, homeDataDirOverride);
        string outputPath =
          dir + "/" +
          VulkanTuner::defaultFileName(gpuName, tuneKey, nnXLen, nnYLen, modelDesc, pipelines->usingFP16Storage);
        VulkanTuneParams::save(outputPath, results);
        if(logger != nullptr)
          logger->write("Done re-tuning Vulkan Winograd transforms, saved results to " + outputPath);
      }
    } else {
      tunedParamsByKey[tuneKey] = pipelines->tuneParams;
      if(logger != nullptr)
        logger->write("Vulkan Winograd transform re-tune did not find valid candidates; keeping loaded params.");
    }
  }

  gpuIdxsNeedingWinogradTransformsTune.clear();
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
  size_t maskElts, maskSumElts, trunkElts;
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
  VBuf trunk;
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
    trunkElts = static_cast<size_t>(model.trunk.trunkNumChannels) * B * paddedSpatialSize;
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
      trunk = makeDeviceBuf(dev, memProps, trunkElts, fp16);
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
        &trunk, persistentBytes(trunkElts, fp16), persistentUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
  const string name;
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
    const VulkanLayerContext& ctx, const ModelDesc* desc, int nnX, int nnY, int paddedSpatialSize_, bool useFP16,
    bool externalInputsUseNhwc_)
    : name(desc->name),
      modelVersion(desc->modelVersion),
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
        VulkanKernels::ExtractChannel0Nhwc::dispatch(
          ctx, extractChannel0NhwcKernel, input, mask, pc, maxBatchSize);
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
        VulkanKernels::ExtractChannel0Nhwc::dispatch(
          ctx, extractChannel0NhwcKernel, nhwcInput, mask, pc, maxBatchSize);
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
    VulkanBuffer* trunkBuf,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership) const {
    (void)trunkBuf;
    // Spatial layout is owned by the model, rather than by individual layers.
    // Only the public NCHW boundaries get a conversion.
    SizedBuf<VulkanBuffer*> convertedInput(
      scratch->allocator.get(), scratch->getBufSizeXY(numInputChannelsPhysical));
    VulkanBuffer* nativeInput = input;
    if(!externalInputsUseNhwc) {
      VulkanKernels::NchwToNhwc::PC pc = {
        numInputChannels, numInputChannelsPhysical, paddedSpatialSize, maxBatchSize};
      VulkanKernels::NchwToNhwc::dispatch(ctx, nchwToNhwcKernel, input, convertedInput.buf, pc);
      cmdComputeBarrier(ctx.cmd, convertedInput.buf->buffer);
      nativeInput = convertedInput.buf;
    }
    SizedBuf<VulkanBuffer*> nativeTrunk(scratch->allocator.get(), scratch->getBufSizeXY(trunkNumChannels));
    trunk->dispatch(ctx, scratch, nativeInput, inputGlobal, inputMeta, nativeTrunk.buf, mask, maskSum, maxBatchSize);
    // trunk->dispatch barriers its own trunkBuf output (trunk tip), so the heads read it directly.
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
  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t userInputMetaBufferElts;

  std::unique_ptr<float[]> userInputBuffer;
  std::unique_ptr<float[]> userInputGlobalBuffer;
  std::unique_ptr<float[]> userInputMetaBuffer;

  std::unique_ptr<float[]> policyPassResults;
  std::unique_ptr<float[]> policyResults;
  std::unique_ptr<float[]> valueResults;
  std::unique_ptr<float[]> scoreValueResults;
  std::unique_ptr<float[]> ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;

    singleInputElts = static_cast<size_t>(m.numInputChannels) * nnXLen * nnYLen;
    singleInputGlobalElts = static_cast<size_t>(m.numInputGlobalChannels);
    singleInputMetaElts = static_cast<size_t>(m.numInputMetaChannels);
    singlePolicyPassResultElts = static_cast<size_t>(m.numPolicyChannels);
    singlePolicyResultElts = static_cast<size_t>(m.numPolicyChannels) * nnXLen * nnYLen;
    singleValueResultElts = static_cast<size_t>(m.numValueChannels);
    singleScoreValueResultElts = static_cast<size_t>(m.numScoreValueChannels);
    singleOwnershipResultElts = static_cast<size_t>(m.numOwnershipChannels) * nnXLen * nnYLen;

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);

    userInputBufferElts = static_cast<size_t>(m.numInputChannels) * maxBatchSize * nnXLen * nnYLen;
    userInputGlobalBufferElts = static_cast<size_t>(m.numInputGlobalChannels) * maxBatchSize;
    userInputMetaBufferElts = static_cast<size_t>(m.numInputMetaChannels) * maxBatchSize;

    userInputBuffer = std::make_unique<float[]>(userInputBufferElts);
    userInputGlobalBuffer = std::make_unique<float[]>(userInputGlobalBufferElts);
    userInputMetaBuffer = std::make_unique<float[]>(userInputMetaBufferElts);

    policyPassResults = std::make_unique<float[]>(static_cast<size_t>(maxBatchSize) * m.numPolicyChannels);
    policyResults =
      std::make_unique<float[]>(static_cast<size_t>(maxBatchSize) * m.numPolicyChannels * nnXLen * nnYLen);
    valueResults = std::make_unique<float[]>(static_cast<size_t>(maxBatchSize) * m.numValueChannels);
    scoreValueResults = std::make_unique<float[]>(static_cast<size_t>(maxBatchSize) * m.numScoreValueChannels);
    ownershipResults =
      std::make_unique<float[]>(static_cast<size_t>(maxBatchSize) * nnXLen * nnYLen * m.numOwnershipChannels);
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
  int nnXLen, nnYLen, policySize;
  int maxBatchSize;
  bool inputsUseNHWC;
  bool requireExactNNLen;
  bool profileBatchSizes;
  std::vector<uint64_t> batchSizeCounts;

  // Per-dispatch GPU-timestamp profiler. Active iff KATAGO_VULKAN_PROFILE_KERNELS=1
  // and the device's compute queue supports timestamps. The reset and timestamp
  // writes are baked into the pre-recorded command buffer at construction time;
  // readback() + printReport() run per submit after vkWaitForFences.
  DispatchProfiler profiler;

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
      policySize(NNPos::getPolicySize(ctx->nnXLen, ctx->nnYLen)),
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

    VulkanLayerContext layerCtx(
      dev->device,
      transferCtx.queue,
      *transferCtx.mutex,
      transferCtx.commandPool,
      dev->info.memoryProperties,
      pipelines->tuneParams,
      dev->info.supportsSubgroupShuffleCompute,
      dev->info.subgroupSize,
      dev->info.canRequireComputeSubgroupSize(dev->info.subgroupSize),
      dev->info.supportsFP16Compute,
      dev->info.supportsDot2F16,
      dev->info.supportsDot2F16AccF16,
      dev->info.supportsCoopmat1F16,
      dev->info.supportsCoopmat1F16AccF16,
      dev->info.supportsCoopmatMaintenance1,
      dev->info.supportsCoopmat2F16,
      dev->info.supportsCoopmat2F16AccF16,
      dev->info.coopmatShapes,
      dev->info.coopmatAccF16Shapes,
      dev->info.coopmat2FlexShapes,
      dev->info.coopmat2AccF16FlexShapes,
      dev->info.coopmat2ReservedSharedBytes,
      dev->info.properties.limits.maxComputeSharedMemorySize,
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

    if(DispatchProfiler::envEnabled()) {
      // NHWC layout islands add two repacks around every spatial convolution.
      // 2048 pairs covers those extra dispatches on the deepest current networks.
      profiler.init(device, timestampPeriodNs, computeQueueSupportsTimestamps, 2048);
    }

    for(int b = 1; b <= maxBatchSize; b++)
      preRecordCommandBufferFor(b, perBatchCmd[b]);
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
      buffers->trunk.get(),
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
    if(profiler.active()) {
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
                dst[(n * paddedSpatialSize + xy) * uploadedSpatialChannels + c] =
                  src[xy * numSpatialFeatures + c];
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
          std::fill(dst, dst + spatialUploadBytes / sizeof(half_t), half_float::half_cast<half_t>(0.0f));
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
// DispatchProfiler — KATAGO_VULKAN_PROFILE_KERNELS instrumentation
// ============================================================================

bool DispatchProfiler::envEnabled() {
  // Read the env var once per process. envEnabled() is called once in the
  // ComputeHandle constructor (before preRecordCommandBuffer) so the result
  // is baked into the pre-recorded command buffer.
  // getenv walks the process env each time — cache to avoid that on the hot path.
  static const bool cached = []() {
    std::string_view v = getenvStringView("KATAGO_VULKAN_PROFILE_KERNELS");
    return !v.empty() && v[0] != '0';
  }();
  return cached;
}

void DispatchProfiler::init(
  VkDevice device_,
  double timestampPeriodNs_,
  bool deviceSupportsTimestamps_,
  uint32_t capacityPairs_) {
  enabled = envEnabled();
  device = device_;
  timestampPeriodNs = timestampPeriodNs_;
  deviceSupportsTimestamps = deviceSupportsTimestamps_;
  capacityPairs = capacityPairs_;
  nextPair = 0;
  if(!enabled || !deviceSupportsTimestamps || capacityPairs == 0)
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
  slots.reserve(capacityPairs);
  // Print the banner once per process (not once per VulkanOutputRun — same process
  // may run many inferences and would otherwise spam stderr).
  static std::atomic<bool> bannerPrinted{false};
  bool expected = false;
  if(bannerPrinted.compare_exchange_strong(expected, true)) {
    std::cerr << "[vk-prof] DispatchProfiler enabled (capacity=" << capacityPairs
              << " pairs, timestampPeriod=" << timestampPeriodNs << "ns)" << std::endl;
  }
}

void DispatchProfiler::destroy() {
  if(pool != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, pool, nullptr);
    pool = VK_NULL_HANDLE;
  }
  slots.clear();
  accums.clear();
  device = VK_NULL_HANDLE;
  enabled = false;
}

void DispatchProfiler::beginCmd(VkCommandBuffer cmd) {
  if(!active())
    return;
  nextPair = 0;
  slots.clear();
  vkCmdResetQueryPool(cmd, pool, 0, capacityPairs * 2);
}

uint32_t DispatchProfiler::recordStart(VkCommandBuffer cmd, std::string_view kernelName, bool fp16) {
  if(!active() || nextPair >= capacityPairs)
    return UINT32_MAX;
  uint32_t idx = nextPair++;
  slots.push_back({kernelName, fp16});
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, idx * 2);
  return idx;
}

void DispatchProfiler::recordEnd(VkCommandBuffer cmd, uint32_t pairIdx) {
  if(pairIdx == UINT32_MAX)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, pairIdx * 2 + 1);
}

void DispatchProfiler::readback() {
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
  std::map<std::string, DispatchProfilerAccum> byName;
  for(uint32_t i = 0; i < nextPair; i++) {
    if(ts[2 * i + 1] <= ts[2 * i])
      continue;
    uint64_t delta = ts[2 * i + 1] - ts[2 * i];
    std::string key = std::string(slots[i].name) + (slots[i].fp16 ? "[fp16]" : "[fp32]");
    auto& a = byName[key];
    a.totalTicks += delta;
    a.count++;
    a.fp16 = slots[i].fp16;
  }
  accums.assign(byName.begin(), byName.end());
  std::sort(accums.begin(), accums.end(), [](const auto& a, const auto& b) {
    return a.second.totalTicks > b.second.totalTicks;
  });
}

void DispatchProfiler::printReport(Logger* logger, int batchSize) const {
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

// ============================================================================
// Section: testEvaluate* Helpers and Entry Points
// ============================================================================
// Helper: create a minimal per-thread context for isolated layer testing.
// testEvaluate* helpers

namespace {

  // Construct a VulkanLayerContext from a work context's compiled pipelines + device.
  VulkanLayerContext makeTestLayerCtx(const VulkanWorkContext& ctx) {
    return VulkanLayerContext(
      ctx.dev->device,
      ctx.dev->computeQueue,
      const_cast<mutex&>(ctx.dev->queueMutex),
      ctx.dev->commandPool,
      ctx.dev->info.memoryProperties,
      ctx.pipelines->tuneParams,
      ctx.dev->info.supportsSubgroupShuffleCompute,
      ctx.dev->info.subgroupSize,
      ctx.dev->info.canRequireComputeSubgroupSize(ctx.dev->info.subgroupSize),
      ctx.dev->info.supportsFP16Compute,
      ctx.dev->info.supportsDot2F16,
      ctx.dev->info.supportsDot2F16AccF16,
      ctx.dev->info.supportsCoopmat1F16,
      ctx.dev->info.supportsCoopmat1F16AccF16,
      ctx.dev->info.supportsCoopmatMaintenance1,
      ctx.dev->info.supportsCoopmat2F16,
      ctx.dev->info.supportsCoopmat2F16AccF16,
      ctx.dev->info.coopmatShapes,
      ctx.dev->info.coopmatAccF16Shapes,
      ctx.dev->info.coopmat2FlexShapes,
      ctx.dev->info.coopmat2AccF16FlexShapes,
      ctx.dev->info.coopmat2ReservedSharedBytes,
      ctx.dev->info.properties.limits.maxComputeSharedMemorySize,
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
            dst[(n * paddedSpatialSize + xy) * numChannels + c] =
              data[(n * nnXYLen + xy) * numChannels + c];
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
    auto kernels = VulkanKernels::NchwToNhwc::build(
      ctx.dev->device, ctx.pipelines->pipelineCache, fp16, ctx.pipelines->tuneParams);
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
            dst[(n * xyLen + xy) * numChannels + c] =
              (float)src[(n * paddedLen + xy) * numChannels + c];
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

  auto inputBuf = useNHWC
                    ? makeTestSpatialNhwcInputBuf(
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

  auto inputBuf = useNHWC
                    ? makeTestSpatialNhwcInputBuf(
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
      CmdCtx{ctx.cmd, pushFn},
      &scratch,
      inputBuf.get(),
      outputBuf.get(),
      maskBuf.get(),
      paddedSpatialSize,
      batchSize);
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

  downloadTestNhwcAsNchwBuf(
    ctx, trunkBuf.get(), outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
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

  downloadTestNhwcAsNchwBuf(
    ctx, trunkBuf.get(), outputBuffer, batchSize, trunkC, nnXYLen, paddedSpatialSize, fp16);
  return true;
}

#endif  // USE_VULKAN_BACKEND

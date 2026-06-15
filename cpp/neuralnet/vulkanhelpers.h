#ifndef NEURALNET_VULKAN_HELPERS_H_
#define NEURALNET_VULKAN_HELPERS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/simpleallocator.h"
#include "../external/half-2.2.0/include/half.hpp"
#include "../neuralnet/vulkanincludes.h"

static_assert(sizeof(half_float::half) == sizeof(uint16_t), "half_float::half must be exactly 16 bits");

// Check a VkResult and throw StringError on failure.
#define VK_CHECK(result) VulkanHelpers::checkResult((result), __FILE__, __LINE__)

// Minimal non-owning view (the project is C++17, no std::span). Constructible
// at compile time from a static constexpr std::array or C array, and at runtime
// from a (pointer, length) pair or a std::vector (e.g. a block wrapper's
// dynamically-built param list). The view does not own its storage; the backing
// buffer must outlive the view.

template<typename T>
struct ArrayView {
  const T* data = nullptr;
  size_t len = 0;
  constexpr ArrayView() = default;
  constexpr ArrayView(const T* data_, size_t len_) : data(data_), len(len_) {}
  template<size_t N>
  constexpr ArrayView(const std::array<T, N>& a) : data(a.data()), len(N) {}
  template<size_t N>
  constexpr ArrayView(const T (&a)[N]) : data(a), len(N) {}
  ArrayView(const std::vector<T>& v) : data(v.data()), len(v.size()) {}
  const T* begin() const { return data; }
  const T* end() const { return data + len; }
  size_t size() const { return len; }
  const T& operator[](size_t i) const { return data[i]; }
};

// Device info

// A usable KHR cooperative-matrix shape. The shader spec constants TM/TN/TK
// must exactly match one of these.
struct CoopmatShape {
  uint32_t m;  // MSize
  uint32_t n;  // NSize
  uint32_t k;  // KSize
  bool operator==(const CoopmatShape& o) const { return m == o.m && n == o.n && k == o.k; }
};

// A usable NV cooperative-matrix2 flexible-dimension record for workgroup-scope
// matrix multiplies.
struct Coopmat2FlexShape {
  uint32_t mGranularity;
  uint32_t nGranularity;
  uint32_t kGranularity;
  uint32_t workgroupInvocations;
  bool operator==(const Coopmat2FlexShape& o) const {
    return mGranularity == o.mGranularity && nGranularity == o.nGranularity && kGranularity == o.kGranularity &&
           workgroupInvocations == o.workgroupInvocations;
  }
};

struct VulkanDeviceInfo {
  int gpuIdx;
  VkPhysicalDevice physicalDevice;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memoryProperties;
  int computeQueueFamilyIdx;
  int transferQueueFamilyIdx;
  int computeQueueCount;
  bool supportsFP16Storage;                   // storageBuffer16BitAccess feature present and enabled
  bool supportsFP16Compute;                   // shaderFloat16 feature present and enabled
  bool supportsShaderRoundingModeRTEFloat16;  // shaderRoundingModeRTEFloat16 property
  bool useShaderRoundingModeRTEFloat16;       // RTE SPIR-V variants selected for this initialized device
  bool shaderFloatControlsNeedsExt;           // API < 1.2 -> must enable VK_KHR_shader_float_controls
  // Physical-device FP16 base support before runtime precision/device-creation
  // choices are applied. Diagnostics only; runtime selection uses supports*.
  bool detectedSupportsFP16Storage;
  bool detectedSupportsFP16Compute;
  bool supportsDot2F16;        // VK_VALVE_shader_mixed_float_dot_product + shaderMixedFloatDotProductFloat16AccFloat32
  bool supportsDot2F16AccF16;  // VK_VALVE_shader_mixed_float_dot_product + Float16AccFloat16
  // The following unfiltered* fields capture optional GEMM acceleration support
  // before runtime env filters and device-creation choices are applied. They
  // are for diagnostics only; runtime selection must use the supports* fields.
  bool unfilteredSupportsDot2F16;
  bool unfilteredSupportsDot2F16AccF16;
  bool unfilteredSupportsCoopmat1F16;
  bool unfilteredSupportsCoopmat1F16AccF16;
  bool unfilteredSupportsCoopmat2F16;
  bool unfilteredSupportsCoopmat2F16AccF16;
  size_t unfilteredCoopmatShapeCount;
  size_t unfilteredCoopmatAccF16ShapeCount;
  size_t unfilteredCoopmat2FlexShapeCount;
  size_t unfilteredCoopmat2AccF16FlexShapeCount;
  bool filterDisabledDot2F16;
  bool filterDisabledDot2F16AccF16;
  bool filterDisabledCoopmat1F16;
  bool filterDisabledCoopmat1F16AccF16;
  bool filterDisabledCoopmat2F16;
  bool filterDisabledCoopmat2F16AccF16;
  // Common base required by both coopmat1 and coopmat2: KHR cooperative matrix
  // feature, FP16 storage/compute base, Vulkan memory model, and compute-stage
  // support. Variant-specific gates are tracked separately below.
  bool supportsKhrCoopmatBase;
  // VK_KHR_cooperative_matrix, f16xf16->f32 at subgroup scope, with the shared
  // FP16 gate (storage + shaderFloat16) and a subgroup-size guarantee. This is
  // the post-device-create "actually enabled and usable" value.
  bool supportsCoopmat1F16;
  // Every usable f16/f16->f32/f32 subgroup-scope shape the device reports. The
  // tuner draws candidate TM/TN/TK from this list and validate() rejects any
  // (TM,TN,TK) not present here. Empty iff supportsCoopmat1F16 is false.
  std::vector<CoopmatShape> coopmatShapes;
  // VK_KHR_cooperative_matrix, f16xf16->f16 at subgroup scope. Kept separate
  // from supportsCoopmat1F16 so FP16-accum variants can tune independently.
  bool supportsCoopmat1F16AccF16;
  std::vector<CoopmatShape> coopmatAccF16Shapes;
  // VK_EXT_cooperative_matrix_maintenance1: queried independently of the
  // bundled Vulkan-Headers version. True only when the three operations used
  // by the experimental attention shader are advertised.
  bool supportsCoopmatMaintenance1;
  // VK_NV_cooperative_matrix2, f16xf16->f32 at workgroup scope. This does NOT
  // inherit coopmat1's subgroup-size-control requirements.
  bool supportsCoopmat2F16;
  uint32_t coopmat2MaxWorkgroupSize;
  uint32_t coopmat2MaxFlexDimension;
  uint32_t coopmat2ReservedSharedBytes;
  std::vector<Coopmat2FlexShape> coopmat2FlexShapes;
  // VK_NV_cooperative_matrix2, f16xf16->f16 at workgroup scope.
  bool supportsCoopmat2F16AccF16;
  std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes;
  // True when KATAGO_VULKAN_ACCEL_VARIANTS constrained optional acceleration
  // families/accumulators during discovery.
  bool gemmVariantFilterConfigured;
  // Canonical no-spaces signature for the selection policy and current GEMM
  // variant env filter.
  std::string gemmVariantFilterSignature;
  // The coopmat shaders declare the Vulkan memory model (via
  // GL_KHR_memory_scope_semantics → OpCapability VulkanMemoryModel), so the
  // vulkanMemoryModel feature must be enabled at device creation for coopmat
  // pipelines to be valid. True iff the device reports the feature; folded into
  // the supportsKhrCoopmatBase gate. On API < 1.2 the VK_KHR_vulkan_memory_model
  // extension must also be enabled (tracked by coopmatNeedsMemModelExt).
  bool supportsVulkanMemoryModel;
  bool coopmatNeedsMemModelExt;               // API < 1.2 → must enable VK_KHR_vulkan_memory_model
  bool supportsPushDescriptors;               // VK_KHR_push_descriptor
  bool supportsSubgroupShuffleCompute;        // compute stage supports subgroup shuffle ops
  uint32_t subgroupSize;                      // reported subgroup size (0 if unsupported/unknown)
  bool supportsSubgroupSizeControl;           // subgroupSizeControl feature enabled path available
  bool supportsComputeFullSubgroups;          // computeFullSubgroups feature available
  bool supportsRequiredSubgroupSizeCompute;   // requiredSubgroupSizeStages includes COMPUTE
  uint32_t minSubgroupSize;                   // 0 if unknown / unsupported
  uint32_t maxSubgroupSize;                   // 0 if unknown / unsupported
  bool hasSubgroupSizeControlExtension;       // VK_EXT_subgroup_size_control listed
  bool supportsPipelineExecutableProperties;  // VK_KHR_pipeline_executable_properties + pipelineExecutableInfo
  bool isIntegrated;
  int defaultDesirability;  // higher = preferred

  bool canRequireComputeSubgroupSize(uint32_t requiredSubgroupSize) const {
    return requiredSubgroupSize > 0 && supportsSubgroupSizeControl && supportsComputeFullSubgroups &&
           supportsRequiredSubgroupSizeCompute && minSubgroupSize > 0 && maxSubgroupSize >= minSubgroupSize &&
           requiredSubgroupSize >= minSubgroupSize && requiredSubgroupSize <= maxSubgroupSize;
  }

  static std::vector<VulkanDeviceInfo> getAllDeviceInfosOnSystem(VkInstance instance, Logger* logger);
};

// Eligibility predicate for the shuffle-based RMSNorm shader variants. Correctness
// of those shaders needs, at once:
//   - subgroup shuffle support (compute stage);
//   - the pipeline can pin the reported subgroup size + require full subgroups
//     (VK_EXT_subgroup_size_control + computeFullSubgroups), so that the
//     spec-defined dense index gl_SubgroupID*gl_SubgroupSize+gl_SubgroupInvocationID
//     covers the entire workgroup;
//   - localSizeX % subgroupSize == 0 (full-subgroups precondition -- otherwise
//     the driver has to give partial subgroups, which the shuffle tree reduce
//     is not correct for);
//   - subgroupSize % wgChannelGroup == 0 (the shuffle group must sit within one
//     subgroup; pass wgChannelGroup=1 for spatial pass 2 which has no such
//     grouping).
inline constexpr bool canUseRMSNormSubgroupVariant(
  bool supportsSubgroupShuffleCompute,
  bool canRequireReportedSubgroupSize,
  uint32_t subgroupSize,
  uint32_t localSizeX,
  uint32_t wgChannelGroup) {
  if(!supportsSubgroupShuffleCompute) return false;
  if(!canRequireReportedSubgroupSize) return false;
  if(subgroupSize == 0) return false;
  if(localSizeX % subgroupSize != 0) return false;
  if(wgChannelGroup == 0 || subgroupSize % wgChannelGroup != 0) return false;
  return true;
}

// Compile-time checks for canUseRMSNormSubgroupVariant. Cheaper than a proper
// unit test for a pure predicate, and impossible to disable.
static_assert(!canUseRMSNormSubgroupVariant(false, true, 32, 64, 32),
              "shuffle unsupported must reject");
static_assert(!canUseRMSNormSubgroupVariant(true, false, 32, 64, 32),
              "cannot pin subgroup size must reject");
static_assert(!canUseRMSNormSubgroupVariant(true, true, 0, 64, 32),
              "unknown subgroup size must reject");
static_assert(!canUseRMSNormSubgroupVariant(true, true, 32, 16, 1),
              "localSizeX=16 with subgroupSize=32 must reject (T4 spatial tile=16 case)");
static_assert(!canUseRMSNormSubgroupVariant(true, true, 32, 64, 64),
              "shuffle group larger than subgroup must reject");
static_assert(canUseRMSNormSubgroupVariant(true, true, 32, 64, 32),
              "T4 per-position NHWC: subgroup=32, localSize=64, wgC=32");
static_assert(canUseRMSNormSubgroupVariant(true, true, 32, 64, 8),
              "T4 per-position narrow channel group: subgroup=32, localSize=64, wgC=8");
static_assert(canUseRMSNormSubgroupVariant(true, true, 32, 32, 1),
              "T4 spatial tile=32: subgroup=32, localSize=32, wgC=1");
static_assert(canUseRMSNormSubgroupVariant(true, true, 32, 128, 1),
              "T4 spatial tile=128: subgroup=32, localSize=128, wgC=1");
static_assert(!canUseRMSNormSubgroupVariant(true, true, 8, 64, 32),
              "subgroupSize=8 cannot fit wgC=32");
static_assert(canUseRMSNormSubgroupVariant(true, true, 8, 64, 8),
              "hypothetical Intel-class: subgroup=8, localSize=64, wgC=8");

// Initialized device

struct InitializedVulkanDevice {
  struct QueueContext {
    VkQueue queue;
    std::mutex* mutex;
    VkCommandPool commandPool;
  };

  VulkanDeviceInfo info;
  VkDevice device;
  VkQueue computeQueue;
  std::vector<VkQueue> computeQueues;
  std::vector<std::unique_ptr<std::mutex>> extraComputeQueueMutexes;
  mutable std::atomic<uint32_t> nextComputeQueueIdx;
  VkQueue transferQueue;
  VkCommandPool commandPool;
  VkCommandPool transferCommandPool;
  // Vulkan queues require external synchronization when accessed from multiple threads.
  std::mutex queueMutex;
  std::unique_ptr<std::mutex> dedicatedTransferQueueMutex;
  std::mutex* transferQueueMutex;

  QueueContext pickComputeContext() const;
  QueueContext getTransferContext() const;

  InitializedVulkanDevice() = delete;
  InitializedVulkanDevice(const InitializedVulkanDevice&) = delete;
  InitializedVulkanDevice& operator=(const InitializedVulkanDevice&) = delete;
  InitializedVulkanDevice(InitializedVulkanDevice&&) = delete;

  explicit InitializedVulkanDevice(VulkanDeviceInfo info_)
    : info(std::move(info_)),
      device(VK_NULL_HANDLE),
      computeQueue(VK_NULL_HANDLE),
      nextComputeQueueIdx(0),
      transferQueue(VK_NULL_HANDLE),
      commandPool(VK_NULL_HANDLE),
      transferCommandPool(VK_NULL_HANDLE),
      transferQueueMutex(&queueMutex) {}

  ~InitializedVulkanDevice();
};

// Device collection (analogous to OpenCL DevicesContext)

struct VulkanDevicesContext {
  int defaultGpuIdx;
  std::vector<std::unique_ptr<InitializedVulkanDevice>> devicesToUse;
  std::vector<std::string> uniqueDeviceNamesToUse;

  VulkanDevicesContext(
    VkInstance instance,
    const std::vector<VulkanDeviceInfo>& allDeviceInfos,
    const std::vector<int>& gpuIdxsToUse,
    bool useFP16Storage,
    bool useFP16Compute,
    Logger* logger);
  ~VulkanDevicesContext();

  VulkanDevicesContext() = delete;
  VulkanDevicesContext(const VulkanDevicesContext&) = delete;
  VulkanDevicesContext& operator=(const VulkanDevicesContext&) = delete;

  const InitializedVulkanDevice* findGpuExn(int gpuIdx) const;
};

// Buffer helpers

// RAII wrapper around a VkBuffer + its VkDeviceMemory.
struct VulkanBuffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  bool ownsMemory;
  VkDeviceSize size;
  VkDevice device;

  VulkanBuffer() : buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE), ownsMemory(true), size(0), device(VK_NULL_HANDLE) {}
  VulkanBuffer(VkBuffer b, VkDeviceMemory m, VkDeviceSize s, VkDevice d, bool ownsMemory_ = true)
    : buffer(b), memory(m), ownsMemory(ownsMemory_), size(s), device(d) {}

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  VulkanBuffer(VulkanBuffer&& o) noexcept
    : buffer(o.buffer), memory(o.memory), ownsMemory(o.ownsMemory), size(o.size), device(o.device) {
    o.buffer = VK_NULL_HANDLE;
    o.memory = VK_NULL_HANDLE;
    o.ownsMemory = true;
  }

  ~VulkanBuffer() {
    if(device != VK_NULL_HANDLE) {
      if(buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, buffer, nullptr);
      if(memory != VK_NULL_HANDLE && ownsMemory)
        vkFreeMemory(device, memory, nullptr);
    }
  }

  bool valid() const { return buffer != VK_NULL_HANDLE; }
};

using VBuf = std::unique_ptr<VulkanBuffer>;

// VulkanHelpers namespace

namespace VulkanHelpers {
  void checkResult(VkResult result, const char* file, int line);

  uint32_t
  findMemoryType(const VkPhysicalDeviceMemoryProperties& memProps, uint32_t typeBits, VkMemoryPropertyFlags required);

  VulkanBuffer allocateBuffer(
    VkDevice device,
    const VkPhysicalDeviceMemoryProperties& memProps,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memFlags);

  // Allocate device-local buffer and upload data from host via a transient command buffer.
  VulkanBuffer allocateAndUploadBuffer(
    VkDevice device,
    VkQueue queue,
    std::mutex& queueMutex,
    VkCommandPool commandPool,
    const VkPhysicalDeviceMemoryProperties& memProps,
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage);

  // Helper: round up to a multiple.
  inline size_t roundUpToMultiple(size_t size, size_t ofThis) {
    return ((size + ofThis - 1) / ofThis) * ofThis;
  }

  // Place a compute→compute buffer memory barrier (write-after-write / read-after-write).
  void cmdComputeBarrier(VkCommandBuffer cmd, VkBuffer buf);

  // Place a compute→compute execution-only ordering barrier on a specific buffer.
  // srcStageMask=COMPUTE / dstStageMask=COMPUTE enforce the execution dependency:
  // prior compute finishes before subsequent compute starts, preventing the GPU
  // scheduling a new writer before the current user of this buffer finishes.
  // Zero access masks — no cache flush. For a scratch slot that will be completely
  // overwritten by the next writer, cache coherency is not needed, only ordering.
  void cmdComputeWARBarrier(VkCommandBuffer cmd, VkBuffer buf);

  // Place a transfer->compute buffer barrier for one buffer.
  void cmdTransferToComputeBarrier(VkCommandBuffer cmd, VkBuffer buf);

  // Place a host->compute buffer barrier for one buffer.
  void cmdHostToComputeBarrier(VkCommandBuffer cmd, VkBuffer buf);

  // Place a compute->transfer buffer barrier for one buffer.
  void cmdComputeToTransferBarrier(VkCommandBuffer cmd, VkBuffer buf);

  // Create a VkShaderModule from precompiled SPIR-V words.
  VkShaderModule createShaderModule(VkDevice device, const uint32_t* spirvData, size_t spirvWords);

  // Opt-in pipeline executable statistics, controlled by KATAGO_VULKAN_PIPELINE_STATS=1.
  bool pipelineExecutableStatsRequested();
  void registerPipelineExecutableStatsDevice(VkDevice device, bool enabled);
  void unregisterPipelineExecutableStatsDevice(VkDevice device);
  bool pipelineExecutableStatsEnabled(VkDevice device);
  void dumpPipelineExecutableStats(VkDevice device, VkPipeline pipeline, std::string_view label);

  // Print all Vulkan physical devices to stdout.
  void printDevices(VkInstance instance, Logger* logger);
}  // namespace VulkanHelpers

// Inline numeric helpers (shared by vulkanbackend.cpp, vulkankernels.cpp,
// vulkanhelpers.cpp).

inline uint32_t divUpU32(int x, int d) {
  return (uint32_t)((x + d - 1) / d);
}

// Round x up to the next multiple of m. Shared by ConvLayer / applyConv (in
// vulkankernels) and the network-composition code (in vulkanbackend.cpp).
inline int roundUpToMultipleInt(int x, int m) {
  return ((x + m - 1) / m) * m;
}

// Buffer allocation / weight upload / scratch infrastructure

// Allocates many buffers in a deferred batch so allocations can be coalesced.
// Call add() for each request, then build() to perform all VkAllocateMemory
// calls grouped by memory type, populating the VBuf out-pointers in place.
class PlannedBufferAllocator {
 public:
  PlannedBufferAllocator(VkDevice device, const VkPhysicalDeviceMemoryProperties& memProps);
  ~PlannedBufferAllocator();

  PlannedBufferAllocator() = delete;
  PlannedBufferAllocator(const PlannedBufferAllocator&) = delete;
  PlannedBufferAllocator& operator=(const PlannedBufferAllocator&) = delete;

  void add(VBuf* dst, VkDeviceSize sizeBytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags);
  void build();

 private:
  struct Request {
    VBuf* dst;
    VkDeviceSize sizeBytes;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags memFlags;
  };

  static VkDeviceSize alignUp(VkDeviceSize x, VkDeviceSize a);
  VkBuffer createBuffer(VkDeviceSize sizeBytes, VkBufferUsageFlags usage) const;

  VkDevice device;
  const VkPhysicalDeviceMemoryProperties& memProps;
  std::vector<Request> requests;
  std::vector<VkDeviceMemory> slabMemories;
  bool built;
};

// Batches weight uploads through one staging buffer and one command buffer per flush.
// Construct one in the thread doing model build, wrap it in WeightUploadBatchGuard,
// and makeWeightBuf() will pick it up automatically.
struct WeightUploadBatch {
  VkDevice device;
  VkQueue queue;
  std::mutex& queueMutex;
  VkCommandPool commandPool;
  VBuf staging;
  void* stagingMapped;
  VkDeviceSize stagingCapacity;
  VkDeviceSize stagingUsed;
  VkCommandBuffer commandBuffer;
  VkFence fence;
  bool hasPendingCopies;

  WeightUploadBatch(
    VkDevice device_,
    VkQueue queue_,
    std::mutex& queueMutex_,
    VkCommandPool commandPool_,
    const VkPhysicalDeviceMemoryProperties& memProps);
  ~WeightUploadBatch();

  WeightUploadBatch() = delete;
  WeightUploadBatch(const WeightUploadBatch&) = delete;
  WeightUploadBatch& operator=(const WeightUploadBatch&) = delete;

  void beginRecording();
  void submitAndWaitIfNeeded();
  VulkanBuffer uploadData(
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const VkPhysicalDeviceMemoryProperties& memProps);
  void finish();
};

// RAII installer for the thread-local current WeightUploadBatch pointer.
// makeWeightBuf reads from that pointer (a static thread_local inside
// vulkanhelpers.cpp); this guard is the only public way to set it.
class WeightUploadBatchGuard {
 public:
  explicit WeightUploadBatchGuard(WeightUploadBatch* batch);
  ~WeightUploadBatchGuard();

  WeightUploadBatchGuard() = delete;
  WeightUploadBatchGuard(const WeightUploadBatchGuard&) = delete;
  WeightUploadBatchGuard& operator=(const WeightUploadBatchGuard&) = delete;
};

// Scratch buffer pool for intermediate activations during inference.
struct ScratchBuffers {
  VkDevice device;
  VkPhysicalDeviceMemoryProperties memProps;
  bool fp16;
  int paddedNNXYLen;
  int maxBatchSize;
  std::unique_ptr<SimpleAllocator<VulkanBuffer*>> allocator;
  size_t batchXYBytes, batchXYFloatBytes, batchFloatBytes, batchBytes;

  // Pre-allocated permanent scratch slots sized by the model's block stack.
  // slot[0] = Winograd input-transformed tiles (also RMSNorm pass1 partials)
  // slot[1] = Winograd post-GEMM tiles          (also RMSNorm pass2 scalar)
  // All blocks share these via element-wise max sizing; they execute sequentially
  // so reuse is safe (see cmdComputeWARBarrier usage in applyConv / RMSNormVk).
  std::vector<VBuf> permanentSlots;

  ScratchBuffers(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp, bool fp16_, int pxyLen, int maxBatch);
  ~ScratchBuffers() = default;

  void setPermanentSlots(const std::vector<VkDeviceSize>& slotSizes);
  VulkanBuffer* permanentSlot(int i) const;
  size_t getBufSizeXY(int c) const { return static_cast<size_t>(c) * batchXYBytes; }
  size_t getBufSizeXYFloat(int c) const { return static_cast<size_t>(c) * batchXYFloatBytes; }
  size_t getBufSizeFloat(int c) const { return static_cast<size_t>(c) * batchFloatBytes; }
  size_t getBufSize(int c) const { return static_cast<size_t>(c) * batchBytes; }

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;
};

// Device-local intermediate buffer of numElts elements (FP16 or FP32).
VBuf makeDeviceBuf(
  VkDevice device,
  const VkPhysicalDeviceMemoryProperties& memProps,
  size_t numElts,
  bool fp16,
  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

VBuf makeDeviceBufFP32(
  VkDevice device,
  const VkPhysicalDeviceMemoryProperties& memProps,
  size_t numElts,
  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

constexpr int WINOGRAD_COOPMAT_PACKED_PAD_WORDS = 2;
constexpr int WINOGRAD_ROW_MAJOR_A_PAD_WORDS = 1;
// Tried coopmat2 packed pad words of 0, 1, 2, and 4; 2 and 4 were equally best,
// so use the smaller 2-word pad.
constexpr int WINOGRAD_COOPMAT2_PACKED_PAD_WORDS = 2;
constexpr int WINOGRAD_COOPMAT_PACKED_B_PAD_WORDS = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
constexpr int STRIDED_COOPMAT_PACKED_B_PAD_SCALARS = 8;
constexpr int STRIDED_DOT2_PACKED_B_PAD_SCALARS = 4;
constexpr int STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS = 8;

size_t winogradCoopmatPackedTileStrideWords(
  int freeDim,
  int k,
  int blockFree,
  int bk,
  int padWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS);
size_t
winogradCoopmatPackedAStrideWords(int m, int k, int bm, int bk, int padWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS);
size_t winogradPackedRowMajorAStrideWords(int m, int k, int bm, int bk, int padWords);
size_t
winogradCoopmatPackedBStrideWords(int n, int k, int bn, int bk, int padWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS);

std::vector<float> packWinogradCoopmatAData(
  const std::vector<float>& data,
  int numBatches,
  int m,
  int k,
  int bm,
  int bk,
  int padWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS);
std::vector<float>
packWinogradRowMajorAData(const std::vector<float>& data, int numBatches, int m, int k, int bm, int bk, int padWords);

std::vector<float> packWinogradCoopmatBWeights(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS);

size_t stridedGemmPackedBStrideVec4s(int n, int k, int bn, int bk, int padScalars);

std::vector<float> packStridedGemmBWeights(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padScalars);
std::vector<float> packStridedGemmBWeightsRowMajor(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padScalars);

// Expand logical 3x3 GEMM weights [9*inChannels,N] so each tap begins at an
// independent tapKStride boundary. Used by direct coopmat2 A tensor loads.
std::vector<float> padConv3x3GemmTapWeights(
  const std::vector<float>& weights,
  int n,
  int inChannels,
  int tapKStride);

// Allocate + upload a weight tensor. Routes through the thread-local
// WeightUploadBatch (installed by WeightUploadBatchGuard) when available;
// otherwise falls back to allocateAndUploadBuffer one tensor at a time.
VBuf makeWeightBuf(
  VkDevice device,
  VkQueue queue,
  std::mutex& queueMutex,
  VkCommandPool commandPool,
  const VkPhysicalDeviceMemoryProperties& memProps,
  const std::vector<float>& weights,
  bool fp16);

#endif  // NEURALNET_VULKAN_HELPERS_H_

#ifndef NEURALNET_VULKAN_BACKEND_H_
#define NEURALNET_VULKAN_BACKEND_H_

#ifdef USE_VULKAN_BACKEND

#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include "../core/commontypes.h"
#include "../core/logger.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkantuner.h"

// Neural net interface implementation for the Vulkan backend.

namespace NeuralNet {

  // Free-function shim around ComputeContext::tuneSelf. Lets vulkantuner.cpp
  // drive the sweep without needing the full ComputeContext type. saveFileOverride
  // empty → write to the per-(gpu,board,model) cache path; non-empty → write to
  // that explicit path.
  void tuneVulkanComputeContext(
    ComputeContext* ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int winograd3x3OutTile,
    int benchIters,
    bool verboseTuner,
    bool full,
    const std::string& saveFileOverride);

}  // namespace NeuralNet

// Defined in vulkanbackend.cpp; only used through pointers here so a forward
// declaration is sufficient for the work-context machinery to live in a header.
struct CompiledPipelines;

// DispatchProfiler
//
// Optional per-dispatch instrumentation for the inference path, in two modes:
//
//   - Timing (KATAGO_VULKAN_PROFILE_KERNELS=1): GPU timestamps around each
//     vkCmdDispatch, accumulated by kernel name and printed as a sorted
//     per-kernel timing breakdown.
//   - Trace (KATAGO_VULKAN_TRACE_KERNELS=1): no timestamps, no query pool —
//     just records which kernels are dispatched and prints the (deduplicated)
//     kernel name set once after the command buffers are pre-recorded. This is
//     for verifying which kernel variant a given config actually runs (e.g. that
//     KATAGO_VULKAN_ACCEL_VARIANTS=dot2 selects the Dot2 kernels), without the
//     timing overhead or verbosity of the full profile.
//
// Both env vars are checked once when the profiler is constructed. When neither
// is set, recordStart/recordEnd return without doing anything — no perf impact
// on shipping inference.
//
// Lifetime: one DispatchProfiler per recording session (e.g. a VulkanOutputRun).
// beginCmd() resets the pool and the slot counter at the start of recording a
// command buffer. Each dispatch records a (start, end) pair around vkCmdDispatch.
// After vkQueueSubmit + fence wait, readback() pulls the timestamps and accumulates
// totals by kernel name. printReport() emits a sorted breakdown.
//
// The pool is sized at construction time; recordStart returns UINT32_MAX (signalling
// "skip recordEnd") when the pool is full, when the profiler is disabled, or when
// the device lacks timestamp support on its compute queue. Trace mode never needs
// the pool, so it works even without timestamp support.

namespace VulkanProfiler {

  struct DispatchProfiler {
    bool profileKernels = false;  // KATAGO_VULKAN_PROFILE_KERNELS timing mode
    bool traceKernels = false;    // KATAGO_VULKAN_TRACE_KERNELS name-only mode
    bool deviceSupportsTimestamps = false;
    double timestampPeriodNs = 1.0;
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t capacityPairs = 0;
    uint32_t nextPair = 0;
    VkDevice device = VK_NULL_HANDLE;

    // Per-dispatch slot metadata (only populated when active).
    struct Slot {
      std::string_view name;
      bool fp16;
    };
    std::vector<Slot> slots;

    // Accumulated totals after readback().
    struct Accum {
      uint64_t totalTicks = 0;
      uint32_t count = 0;
    };
    std::vector<std::pair<std::string, Accum>> accums;

    // True if the environment variable is set and we should be profiling the kernels.
    static bool profileEnvEnabled();
    // True if the environment variable is set and we should be tracing which
    // kernels are being selected by the backend.
    static bool traceEnvEnabled();

    void init(VkDevice device, double timestampPeriodNs, bool deviceSupportsTimestamps, uint32_t capacityPairs);
    void destroy();

    // True when the timing path is live (query pool exists). Trace mode alone
    // records names without timestamps, so readback/printReport must not run on it.
    bool timingActive() const { return profileKernels && deviceSupportsTimestamps && pool != VK_NULL_HANDLE; }
    bool active() const { return timingActive() || traceKernels; }

    // Reset the pool and slot counter for a new recording session.
    void beginCmd(VkCommandBuffer cmd);

    // Record start timestamp + label; returns the pair index for recordEnd. Returns
    // UINT32_MAX if disabled / out of capacity — recordEnd is a no-op in that case.
    uint32_t recordStart(VkCommandBuffer cmd, std::string_view kernelName, bool fp16);
    void recordEnd(VkCommandBuffer cmd, uint32_t pairIdx);

    // After submitAndWait: pull timestamps and accumulate by kernel name.
    void readback();

    // Print a sorted (slowest first) breakdown.
    void printReport(Logger* logger, int batchSize) const;

    // Trace mode: print the deduplicated set of kernels recorded in slots.
    void printTrace(Logger* logger) const;
  };

}  // namespace VulkanProfiler

// VulkanWorkContext
//
// A minimal per-thread Vulkan context: owns a ComputeContext (built on a single
// gpuIdx), a command buffer + fence on the device's compute queue, and an
// optional timestamp query pool. The tuner TuningContext and the layer-test
// TestContext both inherit from this — anything that needs a one-shot Vulkan
// session for recording + timing dispatches.
//
// All member implementations live in vulkanbackend.cpp where ComputeContext
// and CompiledPipelines are full types.

struct VulkanWorkContext {
  ComputeContext* computeCtx = nullptr;  // owned iff ownsComputeCtx
  bool ownsComputeCtx = false;
  const InitializedVulkanDevice* dev = nullptr;
  CompiledPipelines* pipelines = nullptr;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queryPool = VK_NULL_HANDLE;

  bool useGpuTimestamps = false;
  double timestampPeriodNs = 1.0;

  // Allocate a fresh single-GPU ComputeContext (no model-load tuning) and
  // set up command-buffer + fence + optional timestamp pool on its compute
  // queue. destroy() releases all of it including the ComputeContext.
  void init(int gpuIdx, int nnXLen, int nnYLen, enabled_t fp16Mode, Logger* logger);

  // Attach to an existing ComputeContext owned by the caller (not freed by
  // destroy()). Sets up the same command-buffer / fence / timestamp pool on
  // that ComputeContext's device for gpuIdx.
  void initBorrowed(ComputeContext* externalCtx, int gpuIdx, Logger* logger);

  void destroy();

  // Idempotent RAII: releases Vulkan objects and (if owned) the ComputeContext.
  // Ensures that when a VulkanWorkContext is stack-allocated alongside Vulkan-
  // owning locals (buffers, layers), those locals destruct before the device
  // does. Copy/move disabled to keep ownership single-copy.
  ~VulkanWorkContext() { destroy(); }
  VulkanWorkContext() = default;
  VulkanWorkContext(const VulkanWorkContext&) = delete;
  VulkanWorkContext& operator=(const VulkanWorkContext&) = delete;
  VulkanWorkContext(VulkanWorkContext&&) = delete;
  VulkanWorkContext& operator=(VulkanWorkContext&&) = delete;

  VkDevice device() const;
  const VkPhysicalDeviceMemoryProperties& memProps() const;
  PFN_vkCmdPushDescriptorSetKHR getPushDescFn() const;

  void beginRecording();
  void submitAndWait();
  void submitCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size);

  // Upload a flat FP32 array into a fresh device-local FP32 buffer.
  VBuf makeInputBuf(const std::vector<float>& data);
  // Download a flat FP32 device buffer to host floats.
  void downloadFloats(VulkanBuffer* srcBuf, std::vector<float>& dst, size_t numElts);

  // Record `iters` invocations of recordOne and return elapsed seconds.
  double timeDispatches(int iters, const std::function<void()>& recordOne);
  // Record a fixed number of untimed chunks of recordOne invocations.
  void warmupDispatches(int chunkSize, int numChunks, const std::function<void()>& recordOne);

  // Read the FP16 storage/compute modes the CompiledPipelines actually resolved to
  // (the device may downgrade an Auto request). Implemented in vulkanbackend.cpp
  // where CompiledPipelines is a full type. Used by the tuner's TuningContext.
  bool resolvedFP16Storage() const;
  bool resolvedFP16Compute() const;
};

// TuningContext: per-device session the tuner uses.
//
// A VulkanWorkContext plus the FP16 storage/compute flags resolved for the
// device, and FP16-aware upload/download helpers. Lives next to its base in
// vulkanbackend.h so kernel TunableKernel::bench overrides (declared in
// vulkankernels.h via a forward decl) can take it by reference.

namespace VulkanTuner {

  // Full per-candidate bench budget. The tuner's broad screening phases run at
  // a fraction of this (BENCH_SCREEN_TARGET_SECONDS) and restore the full
  // budget for the final confirmation round.
  constexpr double BENCH_TARGET_SECONDS = 1.0;

  struct TuningContext : public VulkanWorkContext {
    bool fp16Storage = false;
    bool fp16Compute = false;
    Logger* logger = nullptr;
    // The autotuner lowers this during broad candidate screening and restores
    // the full budget for the final confirmation round.
    double benchTargetSeconds = BENCH_TARGET_SECONDS;

    // Upload/download in the device's native element type (FP16 or FP32).
    VBuf makeInputBufFP(const std::vector<float>& data);
    void downloadFloatsFP(VulkanBuffer* srcBuf, std::vector<float>& dst, size_t numElts);
  };

  // Internal-only factory used by the standalone Vulkan tuner: builds a
  // ComputeContext bound to the given GPU indices without doing any load-time
  // auto-tuning. The tuner immediately calls ComputeContext::tuneSelf() on the
  // returned context. Not exposed through nninterface.h because no other caller
  // needs an "untuned" ComputeContext.
  ComputeContext* createComputeContextForVulkanTuner(
    const std::vector<int>& gpuIdxs,
    Logger* logger,
    int nnXLen,
    int nnYLen,
    const std::string& homeDataDirOverride,
    enabled_t useFP16Mode);

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
#endif  // NEURALNET_VULKAN_BACKEND_H_

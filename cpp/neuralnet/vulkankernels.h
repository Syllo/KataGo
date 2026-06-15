#ifndef NEURALNET_VULKAN_KERNELS_H_
#define NEURALNET_VULKAN_KERNELS_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanincludes.h"
#include "../neuralnet/vulkantuner.h"

// ComputeKernel: one compute pipeline + its layout.
// Owned by CompiledPipelines (in vulkanbackend.cpp), produced by the per-kernel
// build() factories in namespace VulkanKernels below, dispatched via dispatch().

struct DispatchProfiler;  // defined in vulkanbackend.h

// CmdCtx
//
// Triple of (command buffer, push-descriptor function pointer, optional
// DispatchProfiler) — the three things every dispatch() needs together. Passed
// by const reference through the layer/block tree instead of as separate trailing
// parameters. The profiler is nullptr for most callers; ComputeKernel::dispatch
// checks it once and skips the timestamp pair when null.
struct CmdCtx {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  PFN_vkCmdPushDescriptorSetKHR pushDescFn = nullptr;
  DispatchProfiler* profiler = nullptr;
};

struct LaunchProfile {
  struct None {};
  struct GemmRegN {
    uint16_t regN = 0;
    uint32_t kSpec = 0;
  };
  struct GemmK {
    uint32_t kSpec = 0;
  };
  struct AttentionTiledQ {
    uint16_t qPerThread = 0;
    uint32_t seqLenSpec = 0;
  };
  // DOT2 GEMM tile: BM/BN are needed at dispatch to compute the grid
  // (divUp(M,BM), divUp(N,BN)) since they are tunable, not fixed constants.
  struct GemmDot2Tile {
    uint16_t bm = 0;
    uint16_t bn = 0;
    uint32_t kSpec = 0;
  };

  std::variant<None, GemmRegN, GemmK, AttentionTiledQ, GemmDot2Tile> data = None{};
};

struct ComputeKernel {
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  uint32_t numBindings = 0;

  // Static label (lifetime: program duration), stamped by build() for
  // the optional DispatchProfiler. Identifies the kernel type for the per-kernel
  // breakdown when KATAGO_VULKAN_PROFILE_KERNELS=1. Empty means unnamed.
  std::string_view debugName;
  // FP16 storage flag of the compiled pipeline. Echoed for the profiler's
  // breakdown so e.g. WinogradGemm-fp32 and WinogradGemm-fp16 land in distinct buckets.
  bool debugFp16 = false;

  // Map descriptor-binding index -> index into the call site's "logical buffers"
  // list (e.g. {input, output, scale, bias, mask}). Set when the pipeline knows
  // its FP16/FP32 mode, so the call site doesn't have to re-derive the binding
  // layout per dispatch. If empty, the dispatch overload that takes explicit
  // binding vectors must be used (legacy path; FP32-only helpers).
  std::vector<uint8_t> bindingMap;

  // Echo of the local workgroup dimensions specialised into the shader.
  // Stamped by build() so dispatch() can recompute the grid without taking these
  // values as parameters again. Default 0 means "this kernel doesn't use them";
  // dispatch implementations that read these testAssert(>0) to catch a build()
  // that forgot to stamp them.
  uint16_t localSizeX = 0;
  uint16_t localSizeY = 0;

  // Kernel-specific dispatch semantics that are not actual workgroup dimensions.
  // Kept out of localSizeX/Y to avoid semantic overload.
  LaunchProfile launch;

  ComputeKernel() = default;
  ComputeKernel(const ComputeKernel&) = delete;
  ComputeKernel& operator=(const ComputeKernel&) = delete;

  ComputeKernel(ComputeKernel&& o) noexcept
    : shaderModule(o.shaderModule),
      dsLayout(o.dsLayout),
      pipelineLayout(o.pipelineLayout),
      pipeline(o.pipeline),
      numBindings(o.numBindings),
      debugName(o.debugName),
      debugFp16(o.debugFp16),
      bindingMap(std::move(o.bindingMap)),
      localSizeX(o.localSizeX),
      localSizeY(o.localSizeY),
      launch(std::move(o.launch)) {
    o.shaderModule = VK_NULL_HANDLE;
    o.dsLayout = VK_NULL_HANDLE;
    o.pipelineLayout = VK_NULL_HANDLE;
    o.pipeline = VK_NULL_HANDLE;
    o.numBindings = 0;
    o.debugName = {};
    o.debugFp16 = false;
    o.localSizeX = 0;
    o.localSizeY = 0;
    o.launch.data = LaunchProfile::None{};
  }
  ComputeKernel& operator=(ComputeKernel&& o) noexcept {
    if(this != &o) {
      shaderModule = o.shaderModule;
      dsLayout = o.dsLayout;
      pipelineLayout = o.pipelineLayout;
      pipeline = o.pipeline;
      numBindings = o.numBindings;
      debugName = o.debugName;
      debugFp16 = o.debugFp16;
      bindingMap = std::move(o.bindingMap);
      localSizeX = o.localSizeX;
      localSizeY = o.localSizeY;
      launch = std::move(o.launch);
      o.shaderModule = VK_NULL_HANDLE;
      o.dsLayout = VK_NULL_HANDLE;
      o.pipelineLayout = VK_NULL_HANDLE;
      o.pipeline = VK_NULL_HANDLE;
      o.numBindings = 0;
      o.debugName = {};
      o.debugFp16 = false;
      o.localSizeX = 0;
      o.localSizeY = 0;
      o.launch.data = LaunchProfile::None{};
    }
    return *this;
  }

  void destroy(VkDevice dev);

  // Build a compute pipeline entry from pre-compiled SPIR-V bytes with specialization constants.
  static ComputeKernel build(
    VkDevice device,
    const uint32_t* spirvData,
    size_t spirvWords,
    uint32_t numBindings,
    uint32_t pushConstantBytes,
    const std::vector<VkSpecializationMapEntry>& specEntries,
    const std::vector<uint32_t>& specData,
    VkPipelineCache pipelineCache,
    bool requireFullSubgroups = false,
    uint32_t requiredSubgroupSize = 0,
    std::string_view pipelineStatsLabel = {});

  // Dispatch this pipeline. Uses bindingMap to expand the "logical buffers" list
  // into the full binding list the shader expects (see commentary in vulkankernels.cpp).
  // If ctx.profiler is non-null and active, surrounds vkCmdDispatch with a
  // (start, end) timestamp pair on the profiler's query pool.
  void dispatch(
    const CmdCtx& ctx,
    std::initializer_list<VulkanBuffer*> logicalBufs,
    const void* pushConstantData,
    uint32_t pushConstantSize,
    uint32_t gx,
    uint32_t gy,
    uint32_t gz) const;
};

// Tuning metadata: lightweight, alloc-free descriptions of the tunable spec
// constants a kernel exposes. The tuner (vulkantuner.*) iterates these blind
// to which kernel produced them.

// One tunable spec-constant parameter, identified by a pointer-to-member of
// VulkanTuneParams. The field pointer is BOTH the accessor (cfg.*field) and the
// identity (field == other.field), so there is no name string and no get/set
// lambda. All tunable VulkanTuneParams fields are int32_t, so one pointer type
// addresses any of them.
struct TunableParam {
  int32_t VulkanTuneParams::* field;  // which field this param drives
  ArrayView<int32_t> candidates;      // static "possible tunable values to check"
};

// The values one kernel chose, as (field, value) pairs. Self-applying via the
// field pointer; same-param detection is a pointer-to-member equality check.
struct TunedCandidate {
  struct Entry {
    int32_t VulkanTuneParams::* field;
    int32_t value;
  };
  std::vector<Entry> values;
  double kernelsPerSecond = 0.0;

  void applyTo(VulkanTuneParams& cfg) const {
    for(const auto& e: values)
      cfg.*(e.field) = e.value;
  }
  bool touchesSameParamAs(const TunedCandidate& other) const {
    for(const auto& a: values)
      for(const auto& b: other.values)
        if(a.field == b.field)
          return true;
    return false;
  }
};

// Result of timing one candidate kernel.
struct KernelBench {
  bool ok = false;                // false if the dispatch failed / was invalid
  double kernelsPerSecond = 0.0;  // dispatches per second (higher is better)
  std::vector<float> output;      // flattened result, for RMSE correctness check
};

// TuningContext is the per-device session the tuner uses (defined in
// vulkanbackend.h). Forward-declared so TunableKernel::bench can take it without
// pulling that header in here.
namespace VulkanTuner {
  struct TuningContext;
}

// TunableKernel: polymorphic, kernel-blind unit of tuning.
//
// Concrete kernel classes (in VulkanKernels::, below) derive from this. Each
// kernel is constructed from a VulkanTuneParams (and problem dims), owns the
// ComputeKernel built for that config, and exposes apply() for inference (no
// ComputeKernel argument needed - it owns it). bench() builds representative
// inputs and times its own apply(); validate() rejects configs exceeding device
// limits before bothering to compile.
//
// Meta-kernels (composed network layers/blocks) follow the same shape and own
// child unique_ptr<TunableKernel>s; the virtual destructor tears down the tree.

class TunableKernel {
 public:
  virtual ~TunableKernel() = default;
  virtual std::string name() const = 0;

  // Metadata the tuner iterates blindly: a non-owning view over this kernel's
  // tunable param table (alloc-free). For a micro-kernel this just forwards to
  // its `static sharedParams()` (see below); a block-level meta-kernel instead
  // returns a view over a member vector it assembled in its constructor by
  // concatenating the sharedParams() of the micro-kernels it contains.
  //
  // params() vs sharedParams(): params() is the virtual, instance-bound entry
  // point the tuner calls. sharedParams() is a non-virtual static on each
  // micro-kernel exposing that kernel's param table *without* needing an
  // instance, so meta-kernels can compose tables (e.g. Winograd + WinogradGemm for
  // a residual block) instead of duplicating the candidate-value arrays. A
  // micro-kernel's params() is a one-line `return sharedParams();`.
  virtual ArrayView<TunableParam> params() const = 0;

  // Reject configs exceeding device limits before we bother compiling.
  virtual bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const = 0;

  // Number of compute-shader invocations (threads) per workgroup this config
  // launches. The tuner uses this to prune candidates whose workgroup is
  // narrower than one subgroup: such a workgroup leaves lanes idle every
  // dispatch and is not expected to win the lowest-time selection, so benching
  // it is usually pure waste. Pure virtual so every kernel (present and future)
  // must declare it; return 0 to opt out of the floor when workgroup thread
  // count is not a meaningful axis for the kernel. The tuner only applies the
  // floor when the sweep can otherwise reach a full subgroup, so a kernel whose
  // entire sweep is sub-subgroup by design never has all its candidates pruned.
  virtual uint32_t workgroupThreads(const VulkanTuneParams& cfg) const = 0;

  // Build representative inputs, dispatch + time `iters`. Implementations
  // typically construct a temporary instance for `cfg` and time its apply().
  virtual KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const = 0;

  // Estimated arithmetic work for one dispatch at this kernel instance's
  // problem dimensions. Returns 0.0 when FLOP/s is not a meaningful metric for
  // the kernel. Estimates count the dispatched/padded problem size, matching
  // what the GPU actually runs in the benchmark.
  virtual double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const;
  double estimatedTflops(const VulkanTuneParams& cfg, double kernelsPerSecond) const;

  // How many top candidates to carry forward for a later composite stage.
  // Default 1; big kernels override to ~3.
  virtual int topK() const { return 1; }
};

// Tile multiple required by a kernel's fully optimized path. Tiled kernels have
// looser structural asserts like M % 4 == 0 inline in their dispatch instead.
struct TileAlignmentReq {
  int m = 1;
  int n = 1;
  int k = 1;
};

// Padding the LAYER applies to a problem size before dispatch, declared by
// each kernel variant. This is the "runtime applied padding" contract — NOT
// the kernel's preferred fast-path tile. A kernel returning `{1,1,1,...}` on
// some axis means "the layer applies no padding on this axis"; the shader is
// responsible for guarding its own load/store bounds there.
//
//   - Winograd path: m, n, k are all applied by the layer (numTilesPadded,
//     numOutChannelsPadded, numInChannelsPadded) for the selected GEMM
//     variant, so all three fields are load-bearing.
//   - Strided (1x1 conv, transformer matmul) path: only n is applied by
//     the layer (outChannelsPadded1x1 / outChannelsPadded). M is fixed at
//     paddedSpatialSize (VULKAN_SPATIAL_ALIGN, globally) and K is real
//     (previous layer's outChannels). Strided kernels therefore return
//     m=1 and k=1 regardless of their preferred tile size — those axes
//     are guarded at runtime by the shader itself.
//
// kPaddable is defensive metadata on the K axis. It distinguishes "K tile
// preference, layer must pad" (kPaddable=true) from "K cannot be padded
// across layer boundaries" (kPaddable=false). Strided lcm-merge reads it to
// gate the K-lcm; it is otherwise defensive metadata for future variants.
struct LayerPaddingContract {
  int m = 1;
  int n = 1;
  int k = 1;
  bool kPaddable = true;
};

struct PaddedDims {
  int m;
  int n;
  int k;
};

// Round problem dims up per a layer's padding contract. K is only padded
// when the contract's kPaddable is set — non-paddable axes pass through.
inline PaddedDims padDims(int m, int n, int k, const LayerPaddingContract& c) {
  return {
    roundUpToMultipleInt(m, c.m),
    roundUpToMultipleInt(n, c.n),
    c.kPaddable ? roundUpToMultipleInt(k, c.k) : k,
  };
}

// VulkanKernels: per-kernel build()/dispatch() factories.
// Each struct exposes a nested PC type for push constants and the two statics
// CompiledPipelines / the inference loop call.

namespace VulkanKernels {

  // Backend-wide spatial layout invariant: nnXLen*nnYLen is rounded up to this
  // multiple to produce paddedSpatialSize, which is used as the row stride of
  // spatial buffers. Fixed (not tuned), and must stay a multiple of 8 for the
  // tiled GEMM A-load path.
  constexpr int VULKAN_SPATIAL_ALIGN = 64;
  static_assert(VULKAN_SPATIAL_ALIGN % 8 == 0, "VULKAN_SPATIAL_ALIGN must be divisible by 8 for tiled GEMM access");

  // DOT2 variant default tile config (matching ggml's F16 defaults). These are the
  // SEED defaults for VulkanTuneParams::dot2* / stridedDot2* — the tuner sweeps
  // candidate tiles and the chosen tile is passed to build() at runtime, so these
  // are only the fallback.
  // Invariant (isConfigSupported): BLOCK_SIZE == (BM/WM)*(BN/WN)*WARP so every WM×WN
  // warp region in the BM×BN tile is covered: (64/32)*(64/32)*32 = 128. A smaller
  // value leaves warp_c stuck at 0 and never computes/writes the right half of each
  // N-block.
  constexpr int32_t DOT2_BLOCK_SIZE = 128;
  constexpr int32_t DOT2_BM = 64;
  constexpr int32_t DOT2_BN = 64;
  constexpr int32_t DOT2_WM = 32;
  constexpr int32_t DOT2_WN = 32;
  constexpr int32_t DOT2_WMITER = 2;
  constexpr int32_t DOT2_TM = 4;
  constexpr int32_t DOT2_TN = 2;
  constexpr int32_t DOT2_WARP = 32;
  // K per shared-tile step. Shader-side `#define BK 32` in
  // winograd_gemm_dot2.glsl / gemm_strided_dot2.glsl must match — the shaders
  // rely on this at compile time, not through a spec constant, so cross-check
  // both if you change it.
  constexpr int32_t DOT2_BK = 32;

  // Helpers for assembling the VkSpecializationInfo of a pipeline.
  // Used by per-kernel build() factories AND by the tuner's standalone pipeline
  // builders, so they live in the public header.
  std::vector<VkSpecializationMapEntry> makeSpecMap(int n);
  std::vector<uint32_t> makeSpecData(std::initializer_list<uint32_t> vals);

  struct ScaleBiasMaskAct {
    struct PC {
      int numChannels, paddedSpatialSize, batchSize;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16, int activation);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* scale,
      VulkanBuffer* bias,
      VulkanBuffer* mask,
      const PC& pc);
  };

  struct WinogradTransform : public TunableKernel {
    struct PC {
      int nnXLen, nnYLen, numTilesX, numTilesY, numInChannels, numInChannelsPadded, numTilesReal, numTilesPadded,
        paddedSpatialSize;
    };
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int inTile,
      int outTile,
      int filterSize,
      int offset,
      int localSizeX,
      int localSizeY,
      int packedBM = 64,
      int packedBK = 16,
      int packedAPadWords = WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
    // Device-agnostic build() precondition on the tunable spec values.
    static bool isConfigSupported(int32_t localSizeX, int32_t localSizeY);
    static void
    dispatch(const CmdCtx& ctx, const ComputeKernel& kernel, VulkanBuffer* input, VulkanBuffer* output, const PC& pc);

    // ---- TunableKernel interface ----
    WinogradTransform(int batchSize, int inChannels, int nnXLen, int nnYLen);
    std::string name() const override { return "winogradTransform"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.winogradTransformLocalSizeX * (uint32_t)cfg.winogradTransformLocalSizeY;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemInChannels, problemNnXLen, problemNnYLen;
  };

  struct WinogradBNActTransform : public TunableKernel {
    struct PC {
      int nnXLen, nnYLen, numTilesX, numTilesY, numInChannels, numInChannelsPadded, numTilesReal, numTilesPadded,
        paddedSpatialSize;
    };
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int inTile,
      int outTile,
      int filterSize,
      int offset,
      int activation,
      int localSizeX,
      int localSizeY,
      int packedBM = 64,
      int packedBK = 16,
      int packedAPadWords = WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
    static bool isConfigSupported(int32_t localSizeX, int32_t localSizeY);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* scale,
      VulkanBuffer* bias,
      VulkanBuffer* mask,
      const PC& pc);

    // ---- TunableKernel interface ----
    WinogradBNActTransform(int batchSize, int inChannels, int nnXLen, int nnYLen, int activation);
    std::string name() const override { return "winogradBNActTransform"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.winogradBNActTransformLocalSizeX * (uint32_t)cfg.winogradBNActTransformLocalSizeY;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemInChannels, problemNnXLen, problemNnYLen, problemActivation;
  };

  struct AddPointwise {
    struct PC {
      int numElements;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void
    dispatch(const CmdCtx& ctx, const ComputeKernel& kernel, VulkanBuffer* acc, VulkanBuffer* value, const PC& pc);
  };

  struct AddChannelBiases {
    struct PC {
      int batchChannelCount, paddedSpatialSize;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void
    dispatch(const CmdCtx& ctx, const ComputeKernel& kernel, VulkanBuffer* srcDst, VulkanBuffer* bias, const PC& pc);
  };

  struct AddCBiasActNC {
    struct PC {
      // Word order MUST match add_cbias_act_nc.glsl push constants:
      // {numChannels}. The shader uses numChannels as the channel
      // stride (idx = n*numChannels + c).
      int numChannels;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, int activation);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* data,
      VulkanBuffer* bias,
      const PC& pc,
      int maxBatchSize);
  };

  struct ExtractChannel0 {
    struct PC {
      int numChannels, paddedSpatialSize;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* mask,
      const PC& pc,
      int maxBatchSize);
  };

  struct SumChannels {
    struct PC {
      int paddedSpatialSize;
      int batchSize;
    };
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* mask,
      VulkanBuffer* maskSum,
      const PC& pc,
      int maxBatchSize);
  };

  struct WinogradUntransform : public TunableKernel {
    struct PC {
      int nnXLen, nnYLen, numTilesX, numTilesY, numOutChannels, numOutChannelsPadded, numTilesPadded, paddedSpatialSize;
    };
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int inTile,
      int outTile,
      int filterSize,
      int localSizeX,
      int localSizeY,
      bool addToOutput = false);
    static bool isConfigSupported(int32_t localSizeX, int32_t localSizeY);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      const PC& pc,
      int maxBatchSize);

    // ---- TunableKernel interface ----
    WinogradUntransform(int batchSize, int outChannels, int nnXLen, int nnYLen);
    std::string name() const override { return "winogradUntransform"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.winogradUntransformLocalSizeX * (uint32_t)cfg.winogradUntransformLocalSizeY;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemOutChannels, problemNnXLen, problemNnYLen;
  };

  // WinogradGemm: batched tiled GEMM (Winograd trunk matmul).
  // Used both as a stateless primitive (CompiledPipelines + applyConv call the
  // static build()/dispatch()) and as a TunableKernel: an instance constructed
  // with problem dims (M, N, K, numBatches) participates in the tuner loop.
  // bench() builds a fresh ComputeKernel per candidate cfg internally.
  struct WinogradGemm : public TunableKernel {
    struct PC {
      int M, N, strideA, strideB, strideC;
    };
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int32_t tileM,
      int32_t tileN,
      int32_t tileK,
      int32_t rn,
      int32_t k);
    static bool isConfigSupported(int32_t tileM, int32_t tileN, int32_t tileK, int32_t rn);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches);

    // ---- TunableKernel interface ----
    WinogradGemm(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmTiled"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    // Winograd path: layer pads M to winogradGemmM, N to winogradGemmN, and K to winogradGemmK
    // (numTilesPadded / numOutChannelsPadded / numInChannelsPadded).
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.winogradGemmM, cfg.winogradGemmN, cfg.winogradGemmK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      // Matches validate(): lsx = winogradGemmM/4, lsy = winogradGemmN/winogradGemmRN.
      if(cfg.winogradGemmRN <= 0)
        return 0;
      return (uint32_t)(cfg.winogradGemmM / 4) * (uint32_t)(cfg.winogradGemmN / cfg.winogradGemmRN);
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  struct GemmStridedTiled : public TunableKernel {
    struct PC {
      int M, N, N_real, strideA, strideB, strideC;
    };
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int32_t localSizeX,
      int32_t localSizeY,
      int32_t tileK,
      int32_t rn,
      int32_t k,
      bool addToOutput = false);
    static bool isConfigSupported(int32_t localSizeX, int32_t localSizeY, int32_t tileK, int32_t rn);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches);

    // ---- TunableKernel interface ----
    GemmStridedTiled(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedTiled"; }
    static ArrayView<TunableParam> sharedParams();
    // Strided path: layer applies N-padding only (outChannelsPadded to 8). M is
    // pinned to paddedSpatialSize globally, K is unpaddable. Fast-path tile is
    // {8,8,8} but only n is enforced by the runtime padding contract.
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& /*cfg*/) { return {1, 8, 1, false}; }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.gemmStridedTiledLocalSizeX * (uint32_t)cfg.gemmStridedTiledLocalSizeY;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // WinogradGemmDot2: DOT2 variant using OpFDot2MixAcc32VALVE (SPV_VALVE_mixed_float_dot_product).
  // FP16 only. Uses ggml-style warp tiling with f16vec2 shared memory and v_dot2_f32_f16.
  // Requires supportsDot2F16 && supportsFP16Storage.
  struct WinogradGemmDot2 : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp,
      int32_t k);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches);

    WinogradGemmDot2(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmDot2"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    // Winograd DOT2: layer pads M to dot2BM, N to dot2BN, K to DOT2_BK
    // (numInChannelsPadded is zero-filled by the transform, so K IS paddable).
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.dot2BM, cfg.dot2BN, DOT2_BK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.dot2BlockSize; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  // WinogradGemmDot2AccF16: DOT2 variant using OpFDot2MixAcc16VALVE.
  // Uses separate tuner fields from WinogradGemmDot2 because FP16
  // accumulation can prefer different tile shapes.
  struct WinogradGemmDot2AccF16 : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp,
      int32_t k);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches);

    WinogradGemmDot2AccF16(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmDot2AccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.dot2AccF16BM, cfg.dot2AccF16BN, DOT2_BK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.dot2AccF16BlockSize; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  // GemmStridedDot2: DOT2 variant for 1x1 conv / transformer matmul.
  // FP16 only. Same architecture as WinogradGemmDot2 with N_real guard on C store.
  struct GemmStridedDot2 : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      bool addToOutput = false);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches);

    GemmStridedDot2(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedDot2"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    // Strided DOT2: layer applies N-padding only (to stridedDot2BN). M stays at
    // paddedSpatialSize (globally pinned to VULKAN_SPATIAL_ALIGN, not re-padded
    // per variant), and K stays real. Runtime M and K-tail checks are controlled
    // by ALIGNED and K_ALIGNED respectively.
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedDot2BN, 1, false};
    }
    // DOT2 shader tile shape is {stridedDot2BM, stridedDot2BN, DOT2_BK}.
    // On the strided path M/K are not layer-padded, so runtime ALIGNED gates
    // on M/N only while K_ALIGNED gates the K-tail guard.
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedDot2BM, cfg.stridedDot2BN, DOT2_BK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.stridedDot2BlockSize; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // GemmStridedDot2AccF16: strided DOT2 variant using OpFDot2MixAcc16VALVE.
  struct GemmStridedDot2AccF16 : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      bool addToOutput = false);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t wm,
      int32_t wn,
      int32_t wmIter,
      int32_t tm,
      int32_t tn,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches);

    GemmStridedDot2AccF16(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedDot2AccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedDot2AccF16BN, 1, false};
    }
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedDot2AccF16BM, cfg.stridedDot2AccF16BN, DOT2_BK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.stridedDot2AccF16BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // WinogradGemmCoopmat: coopmat1 (VK_KHR_cooperative_matrix) variant of the
  // Winograd trunk matmul. FP16 storage in, FP32 accumulate, subgroup-scope
  // fragments. Requires supportsCoopmat1F16. TM/TN/TK are the device's reported
  // coopmat fragment shape (MSize/NSize/KSize), NOT free knobs; the tuner sets
  // them from VulkanDeviceInfo::coopmatShapes and bench() rejects any (TM,TN,TK)
  // not present in that list.
  struct WinogradGemmCoopmat : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp,
      int32_t k,
      uint32_t requiredSubgroupSize);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches);

    WinogradGemmCoopmat(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmCoopmat"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    // Winograd coopmat: layer pads M to coopmatBM, N to coopmatBN, K to coopmatBK
    // (numInChannelsPadded is zero-filled by the transform, so K IS paddable).
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.coopmatBM, cfg.coopmatBN, cfg.coopmatBK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.coopmatBlockSize; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  // GemmStridedCoopmat: coopmat1 variant for 1x1 conv / transformer matmul.
  // Same architecture as WinogradGemmCoopmat with N_real guard on C store.
  struct GemmStridedCoopmat : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      uint32_t requiredSubgroupSize,
      bool addToOutput = false);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches);

    GemmStridedCoopmat(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedCoopmat"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    // Strided coopmat: layer applies N-padding only (to stridedCoopmatBN). M stays
    // at paddedSpatialSize, K stays real. ALIGNED and K_ALIGNED control M/N and
    // K-tail checks separately.
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedCoopmatBN, 1, false};
    }
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedCoopmatBM, cfg.stridedCoopmatBN, cfg.stridedCoopmatBK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.stridedCoopmatBlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // FP16-accumulator coopmat1 variants. Same storage/layout ABI as the
  // FP32-accumulator coopmat1 variants, but require f16/f16->f16/f16 shapes.
  struct WinogradGemmCoopmatAccF16 : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp,
      int32_t k,
      uint32_t requiredSubgroupSize);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp) {
      return WinogradGemmCoopmat::isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
    }
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches) {
      WinogradGemmCoopmat::dispatch(ctx, kernel, A, B, C, k, pc, numBatches);
    }

    WinogradGemmCoopmatAccF16(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmCoopmatAccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.coopmatAccF16BM, cfg.coopmatAccF16BN, cfg.coopmatAccF16BK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.coopmatAccF16BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  struct GemmStridedCoopmatAccF16 : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      uint32_t requiredSubgroupSize,
      bool addToOutput = false);
    static bool isConfigSupported(
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t wm,
      int32_t wn,
      int32_t tm,
      int32_t tn,
      int32_t tk,
      int32_t warp) {
      return GemmStridedCoopmat::isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
    }
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches) {
      GemmStridedCoopmat::dispatch(ctx, kernel, input, filter, output, k, pc, numBatches);
    }

    GemmStridedCoopmatAccF16(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedCoopmatAccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedCoopmatAccF16BN, 1, false};
    }
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedCoopmatAccF16BM, cfg.stridedCoopmatAccF16BN, cfg.stridedCoopmatAccF16BK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.stridedCoopmatAccF16BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // WinogradGemmCoopmat2: VK_NV_cooperative_matrix2 workgroup-scope variant.
  // Uses SSBO tensor loads over packed Winograd A/B layouts, FP16 storage in,
  // FP32 accumulate, FP16 output. Requires supportsCoopmat2F16.
  struct WinogradGemmCoopmat2 : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel
    build(VkDevice device, VkPipelineCache cache, int32_t blockSize, int32_t bm, int32_t bn, int32_t bk, int32_t k);
    static bool isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches);

    WinogradGemmCoopmat2(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmCoopmat2"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.coopmat2BlockSize; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  // GemmStridedCoopmat2: VK_NV_cooperative_matrix2 variant for 1x1 conv /
  // transformer matmul. Same C-store single-writer contract as the other
  // strided variants.
  struct GemmStridedCoopmat2 : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      bool addToOutput = false);
    static bool isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches);

    GemmStridedCoopmat2(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedCoopmat2"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedCoopmat2BN, 1, false};
    }
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedCoopmat2BM, cfg.stridedCoopmat2BN, cfg.stridedCoopmat2BK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.stridedCoopmat2BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  // FP16-accumulator coopmat2 variants.
  struct WinogradGemmCoopmat2AccF16 : public TunableKernel {
    using PC = WinogradGemm::PC;
    static ComputeKernel
    build(VkDevice device, VkPipelineCache cache, int32_t blockSize, int32_t bm, int32_t bn, int32_t bk, int32_t k);
    static bool isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
      return WinogradGemmCoopmat2::isConfigSupported(blockSize, bm, bn, bk);
    }
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* A,
      VulkanBuffer* B,
      VulkanBuffer* C,
      int k,
      const PC& pc,
      int numBatches) {
      WinogradGemmCoopmat2::dispatch(ctx, kernel, A, B, C, k, pc, numBatches);
    }

    WinogradGemmCoopmat2AccF16(int m, int n, int k, int numBatches);
    std::string name() const override { return "winogradGemmCoopmat2AccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {cfg.coopmat2AccF16BM, cfg.coopmat2AccF16BN, cfg.coopmat2AccF16BK, true};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.coopmat2AccF16BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK, problemNumBatches;
  };

  struct GemmStridedCoopmat2AccF16 : public TunableKernel {
    using PC = GemmStridedTiled::PC;
    static constexpr bool kSingleWriterStore = true;
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk,
      int32_t aligned,
      int32_t packedB,
      int32_t kAligned,
      int32_t k,
      bool addToOutput = false);
    static bool isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
      return GemmStridedCoopmat2::isConfigSupported(blockSize, bm, bn, bk);
    }
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      int k,
      const PC& pc,
      int numBatches) {
      GemmStridedCoopmat2::dispatch(ctx, kernel, input, filter, output, k, pc, numBatches);
    }

    GemmStridedCoopmat2AccF16(int batchSize, int M, int N, int K);
    std::string name() const override { return "gemmStridedCoopmat2AccF16"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    static LayerPaddingContract layerPaddingContract(const VulkanTuneParams& cfg) {
      return {1, cfg.stridedCoopmat2AccF16BN, 1, false};
    }
    static TileAlignmentReq tileAlignmentReq(const VulkanTuneParams& cfg) {
      return {cfg.stridedCoopmat2AccF16BM, cfg.stridedCoopmat2AccF16BN, cfg.stridedCoopmat2AccF16BK};
    }
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.stridedCoopmat2AccF16BlockSize;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemM, problemN, problemK;
  };

  struct GemmDirectFP32 : public TunableKernel {
    struct PC {
      int M, N;
    };
    static ComputeKernel
    build(VkDevice device, VkPipelineCache cache, int32_t localSizeX, int32_t localSizeY, int32_t k);
    static bool isConfigSupported(int32_t localSizeX, int32_t localSizeY);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* weight,
      VulkanBuffer* output,
      int k,
      const PC& pc);

    // ---- TunableKernel interface ----
    GemmDirectFP32(int M, int N, int K);
    std::string name() const override { return "gemmDirect"; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.gemmDirectLocalSizeX * (uint32_t)cfg.gemmDirectLocalSizeY;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemM, problemN, problemK;
  };

  // Tiled flash-attention. Specialised per (seqLen, qHeadDim, vHeadDim) and tuned over
  // attnBlockQ, attnBlockKV, attnQPerThread.
  struct AttentionTiled : public TunableKernel {
    struct PC {
      int numHeads, numKVHeads;
      float scale;
      int numBH;
    };
    static ComputeKernel build(
      VkDevice device,
      VkPipelineCache cache,
      bool fp16,
      int blockQ,
      int blockKV,
      int qPerThread,
      int seqLen,
      int headDim,
      int vHeadDim,
      bool useRope,
      bool learnableRope);
    static bool isConfigSupported(int32_t blockQ, int32_t blockKV, int32_t qPerThread);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* Q,
      VulkanBuffer* K,
      VulkanBuffer* V,
      VulkanBuffer* output,
      VulkanBuffer* mask,
      VulkanBuffer* cos,
      VulkanBuffer* sin,
      int seqLen,
      const PC& pc);

    AttentionTiled(
      int batchSize,
      int numTokens,
      int headDim,
      int vHeadDim,
      int numHeads,
      int numKVHeads,
      bool useRope,
      bool learnableRope);
    std::string name() const override { return "attnTiled"; }
    int topK() const override { return 3; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.attnBlockQ;  // localSizeX = blockQ, localSizeY = 1
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemNumTokens, problemHeadDim, problemVHeadDim, problemNumHeads, problemNumKVHeads;
    bool problemUseRope, problemLearnableRope;
  };

  struct SwiGLU {
    struct PC {
      int numElements;
    };
    static constexpr uint32_t ELEMENTS_PER_VEC = 4;
    static constexpr uint32_t LOCAL_SIZE_X = 64;
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* main,
      VulkanBuffer* gate,
      VulkanBuffer* output,
      const PC& pc);
  };

  struct GPoolReduction : public TunableKernel {
    struct PC {
      int numChannels, paddedSpatialSize;
    };
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16, int xyStride);
    static bool isConfigSupported(int32_t xyStride);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* mask,
      VulkanBuffer* maskSum,
      const PC& pc,
      int maxBatchSize);

    // ---- TunableKernel interface ----
    GPoolReduction(int batchSize, int channels, int xySize);
    std::string name() const override { return "gpoolReduction"; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.gpoolXystride; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemChannels, problemXySize;
  };

  struct ValueHeadPool : public TunableKernel {
    struct PC {
      int numChannels, paddedSpatialSize;
    };
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16, int xyStride);
    static bool isConfigSupported(int32_t xyStride);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* convOut,
      VulkanBuffer* concat,
      VulkanBuffer* maskSum,
      const PC& pc,
      int maxBatchSize);

    // ---- TunableKernel interface ----
    ValueHeadPool(int batchSize, int channels, int xySize);
    std::string name() const override { return "valueHeadPool"; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override {
      return (uint32_t)cfg.valueHeadPoolXystride;
    }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemChannels, problemXySize;
  };

  // Per-position RMSNorm across channels. Spec constants: WG_XY_SIZE=8,
  // WG_C_SIZE=8, C_PER_THREAD=1, USE_FP16_STORAGE. Grid: gx = divUp(paddedSpatialSize, 8).
  struct TransformerRMSNorm {
    struct PC {
      int numChannels, paddedSpatialSize;
      float epsilon;
    };
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16, bool useSubgroupVariant);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* gamma,
      VulkanBuffer* beta,
      VulkanBuffer* mask,
      const PC& pc,
      int maxBatchSize);
  };

  // Direct (non-Winograd) 2-D convolution for small or non-3x3/5x5 filter sizes.
  // Spec constants: TILE_XSIZE=8, TILE_YSIZE=4, TILE_CHANNELS=4, USE_FP16_STORAGE.
  // Grid: gx = divUp(nnXLen, 8), gy = divUp(nnYLen, 4), gz = numOutChannels.
  struct Conv2dDirect {
    struct PC {
      int nnXLen, nnYLen, numOutChannels, numInChannels, filterXRadius, filterYRadius, paddedSpatialSize, batchSize;
    };
    static ComputeKernel build(VkDevice device, VkPipelineCache cache, bool fp16);
    static void dispatch(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* filter,
      VulkanBuffer* output,
      const PC& pc);
  };

  // Spatial RMSNorm is a three-pass kernel: pass1 = per-tile sum-of-squares,
  // pass2 = final reduction across tiles into a scalar per batch element,
  // pass3 = normalise. Pass1/pass2 share the TILE_SIZE spec constant
  // (the power-of-two reduction-tile knob, here called spatialRMSNormTile).
  // Pass3's APPLY_ELTS_PER_THREAD is pinned at 4 — not tuned. The
  // TunableKernel interface tunes spatialRMSNormTile across the pass1+pass2
  // chain (pass3 timing is included so the search reflects the full RMSNorm
  // cost a tiled kernel would see in inference).
  struct SpatialRMSNorm : public TunableKernel {
    // Top-level PC carried by the orchestrator dispatch(). The three per-pass
    // PCs are derived from this plus the spec'd tileSize (read from kernel.localSizeX).
    struct PC {
      int numChannels, paddedSpatialSize;
      float epsilon;
    };
    struct Pass1PC {
      int numChannels, paddedSpatialSize, tilesPerGroup, numCHWWorkgroups;
    };
    struct Pass2PC {
      int numPartials, tilesPerGroup;
    };
    struct Pass3PC {
      int numChannels, paddedSpatialSize;
      float epsilon;
    };

    static ComputeKernel buildPass1(VkDevice device, VkPipelineCache cache, bool fp16, int tileSize);
    static ComputeKernel buildPass2(VkDevice device, VkPipelineCache cache, int tileSize, bool useSubgroupVariant);
    static ComputeKernel buildPass3(VkDevice device, VkPipelineCache cache, bool fp16);
    static bool isConfigSupported(int32_t tileSize);

    // Unified build: returns the three pass pipelines as one array, in
    // {pass1, pass2, pass3} order. Thin wrapper over buildPass1/2/3 so callers
    // own a single object and dispatch via the matching dispatch() overload.
    static std::array<ComputeKernel, 3>
    build(VkDevice device, VkPipelineCache cache, bool fp16, int tileSize, bool useSubgroupVariant);

    // Each per-pass dispatch derives its grid from its PC and the kernel's
    // stamped localSizeX (== tileSize for pass1/pass2; unused for pass3).
    static void dispatchPass1(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* mask,
      VulkanBuffer* partials,
      const Pass1PC& pc,
      int maxBatchSize);
    static void dispatchPass2(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* partials,
      VulkanBuffer* scalar,
      const Pass2PC& pc,
      int maxBatchSize);
    static void dispatchPass3(
      const CmdCtx& ctx,
      const ComputeKernel& kernel,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* gamma,
      VulkanBuffer* beta,
      VulkanBuffer* mask,
      VulkanBuffer* maskSum,
      VulkanBuffer* scalar,
      const Pass3PC& pc,
      int maxBatchSize);

    // Unified dispatch: records the full three-pass chain (pass1 -> pass2 ->
    // pass3) including the two inter-pass compute barriers. `kernels` is what
    // build() returned. `partials`/`scalar` are the two scratch buffers (pass1
    // partial sums, pass2 per-batch scalar); the caller owns them and issues any
    // trailing barrier on `output`/`partials`/`scalar`. The spec'd tileSize is
    // read from kernels[0].localSizeX.
    static void dispatch(
      const CmdCtx& ctx,
      const std::array<ComputeKernel, 3>& kernels,
      VulkanBuffer* input,
      VulkanBuffer* output,
      VulkanBuffer* gamma,
      VulkanBuffer* beta,
      VulkanBuffer* mask,
      VulkanBuffer* maskSum,
      VulkanBuffer* partials,
      VulkanBuffer* scalar,
      const PC& pc,
      int maxBatchSize);

    // Pass3's apply-loop stride and X local size. Pinned in the shader's
    // defaults and mirrored here so the host grid math agrees with the
    // compiled pipeline.
    static constexpr int APPLY_ELTS_PER_THREAD = 4;
    static constexpr int PASS3_LOCAL_X = 64;

    // ---- TunableKernel interface ----
    SpatialRMSNorm(int batchSize, int channels, int xySize);
    std::string name() const override { return "spatialRMSNorm"; }
    static ArrayView<TunableParam> sharedParams();
    ArrayView<TunableParam> params() const override;
    bool validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const override;
    // pass1/pass2 launch a tileSize-wide workgroup (local_size_x_id = 0). pass3's
    // workgroup is pinned at PASS3_LOCAL_X and not tuned, so the tunable axis is
    // the reduction tile.
    uint32_t workgroupThreads(const VulkanTuneParams& cfg) const override { return (uint32_t)cfg.spatialRMSNormTile; }
    KernelBench bench(VulkanTuner::TuningContext& ctx, const VulkanTuneParams& cfg, int iters) const override;
    double estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const override;

   private:
    int problemBatchSize, problemChannels, problemXySize;
  };

}  // namespace VulkanKernels

// VulkanTuner: search-loop declarations.
//
// Lives here (rather than vulkantuner.h) because these reference TunableKernel
// and TunedCandidate, which are declared above. vulkantuner.h forwards the
// declaration of these in a comment; non-Vulkan TUs (the OpenCL build of
// command/tune.cpp) include only vulkantuner.h and never see these.

#include <memory>
#include <ostream>

namespace VulkanTuner {

  struct TuningContext;  // full def in vulkanbackend.h

  // Build the set of micro-kernels to tune for one model (skips transformer
  // kernels when the model has no transformer blocks). The problem sizes are
  // derived from modelDesc / board / batch, matching the old per-stage setup.
  std::vector<std::unique_ptr<TunableKernel>>
  makeMicroKernels(const ModelDesc* modelDesc, int nnXLen, int nnYLen, int batchSize, const VulkanTuneParams& cfg);

  // Cross-product sweep over one kernel's params; returns the fastest topK()
  // RMSE-passing candidates (sorted fastest first; empty if none pass).
  std::vector<TunedCandidate> tuneOne(
    TunableKernel& kernel,
    const VulkanTuneParams& seed,
    TuningContext& ctx,
    int iters,
    std::ostream& out,
    bool verboseTuner);

  // Top-level: tune every micro-kernel and block-level kernel on the open context,
  // fold the winners into tunedConfig. modelDesc must be non-null.
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
    VulkanTuneParams& tunedConfig);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKAN_KERNELS_H_

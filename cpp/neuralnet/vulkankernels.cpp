#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkankernels.h"
#include <algorithm>
#include <array>
#include "../core/test.h"
#include "../neuralnet/vulkanbackend.h"
// For resolveWinogradGemmVariant / selectedWinogradPaddingContract: the FLOP
// estimator reuses the runtime's variant selection instead of duplicating it.
#include "../neuralnet/vulkanlayers.h"
#include "vulkanshaders_generated.h"

using std::vector;

// ============================================================================
// TU-local helpers
// ============================================================================

// Power-of-two check shared by the reduction kernels' isConfigSupported()
// predicates (their butterfly reduction requires a power-of-two tile size).
namespace {

  bool isPow2(int32_t v) {
    return v > 0 && (v & (v - 1)) == 0;
  }

  // Build a TunableParam whose candidate list comes from the .inc's
  // parenthesized CANDIDATES argument. VULKAN_TUNE_PARAM_UNWRAP strips the
  // parens so the values become the template pack; each instantiation owns a
  // per-static (magic-statics, thread-safe) read-only array, so the returned
  // ArrayView stays valid for the program lifetime.
#define VULKAN_TUNE_PARAM_UNWRAP(...) __VA_ARGS__
  template<int32_t... Vals>
  TunableParam makeTunableParam(int32_t VulkanTuneParams::* field) {
    static const std::array<int32_t, sizeof...(Vals)> storage = {{Vals...}};
    return {field, ArrayView<int32_t>(storage)};
  }

  template<typename T>
  const T& requireLaunchProfile(const ComputeKernel& kernel) {
    const T* profile = std::get_if<T>(&kernel.launch.data);
    testAssert(profile != nullptr);
    return *profile;
  }

  // Fallback for a coopmat-family build() reached when the toolchain did not
  // compile that shader family. Unreachable in practice — the corresponding
  // supports* flag is forced false at detection — so a single assert suffices.
  // Only referenced from #else branches, so [[maybe_unused]] when the coopmat
  // shaders ARE compiled.
  [[maybe_unused]] ComputeKernel failUncompiledCoopmat() {
    testAssert(false && "a coopmat-family build() was called but the toolchain did not compile its shaders");
    return ComputeKernel();
  }

  double gemmFlopsPerDispatch(int batches, int m, int n, int k) {
    if(batches <= 0 || m <= 0 || n <= 0 || k <= 0)
      return 0.0;
    return 2.0 * (double)batches * (double)m * (double)n * (double)k;
  }

  double attentionMatmulFlopsPerDispatch(int batchSize, int seqLen, int headDim, int vHeadDim, int numHeads) {
    if(batchSize <= 0 || seqLen <= 0 || headDim <= 0 || vHeadDim <= 0 || numHeads <= 0)
      return 0.0;
    double batchHeads = (double)batchSize * (double)numHeads;
    double tokenPairs = (double)seqLen * (double)seqLen;
    return 2.0 * batchHeads * tokenPairs * (double)(headDim + vHeadDim);
  }

  int ceilDivInt(int x, int divisor) {
    testAssert(divisor > 0);
    return x <= 0 ? 0 : (x + divisor - 1) / divisor;
  }

  int roundUpToMultiple(int x, int multiple) {
    return ceilDivInt(x, multiple) * multiple;
  }

  uint32_t divUpU32(int x, int d) {
    return (uint32_t)ceilDivInt(x, d);
  }

  LayerPaddingContract selectedWinogradPaddingContractForEstimate(const VulkanTuneParams& cfg) {
    // The FLOP estimator has no device info, so "supported" is implied by
    // "tuned and measured" (unsupported variants are never measured). Passing
    // every support flag as true therefore reproduces the runtime's selection
    // exactly, using the same resolveWinogradGemmVariant implementation rather
    // than a parallel copy of the tier/margin policy.
    const int64_t variant = resolveWinogradGemmVariant(cfg, true, true, true, true, true, true, true);
    return selectedWinogradPaddingContract(cfg, variant);
  }

  double
  winogradTransformInvocationCount(int batchSize, int channels, int nnXLen, int nnYLen, const VulkanTuneParams& cfg) {
    if(batchSize <= 0 || channels <= 0 || nnXLen <= 0 || nnYLen <= 0)
      return 0.0;
    LayerPaddingContract contract = selectedWinogradPaddingContractForEstimate(cfg);
    int outTile = nhwcWinograd3x3OutTileFor(cfg);
    int numTilesX = ceilDivInt(nnXLen, outTile);
    int numTilesY = ceilDivInt(nnYLen, outTile);
    int numTilesPadded = roundUpToMultiple(batchSize * numTilesX * numTilesY, std::max(1, contract.m));
    int kAlignment = contract.kPaddable ? contract.k : 1;
    int channelsPadded = roundUpToMultiple(channels, std::max(1, kAlignment));
    return (double)numTilesPadded * (double)channelsPadded;
  }

  bool isSupportedWinogradGeometry(int filterSize, int inTile, int outTile) {
    return (filterSize == 3 && inTile == 4 && outTile == 2) || (filterSize == 3 && inTile == 6 && outTile == 4) ||
           (filterSize == 5 && inTile == 6 && outTile == 2);
  }

  void requireSupportedWinogradGeometry(std::string_view kernelName, int filterSize, int inTile, int outTile) {
    if(isSupportedWinogradGeometry(filterSize, inTile, outTile))
      return;
    throw StringError(
      std::string(kernelName) + ": unsupported Winograd geometry (filterSize=" + Global::intToString(filterSize) +
      ", inTile=" + Global::intToString(inTile) + ", outTile=" + Global::intToString(outTile) + ")");
  }

  [[maybe_unused]] std::string coopmatPipelineStatsLabel(
    std::string_view name,
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t sgm,
    int32_t sgn,
    int32_t tm,
    int32_t tn,
    int32_t tk,
    int32_t subgroupSize) {
    return std::string(name) + " tile=[bs=" + std::to_string(workgroupSize) + " bm=" + std::to_string(bm) +
           " bn=" + std::to_string(bn) + " bk=" + std::to_string(bk) + " sgm=" + std::to_string(sgm) +
           " sgn=" + std::to_string(sgn) + " tm=" + std::to_string(tm) + " tn=" + std::to_string(tn) +
           " tk=" + std::to_string(tk) + " subgroup=" + std::to_string(subgroupSize) + "]";
  }

  [[maybe_unused]] std::string
  coopmat2PipelineStatsLabel(std::string_view name, int32_t workgroupSize, int32_t bm, int32_t bn, int32_t bk) {
    return std::string(name) + " tile=[bs=" + std::to_string(workgroupSize) + " bm=" + std::to_string(bm) +
           " bn=" + std::to_string(bn) + " bk=" + std::to_string(bk) + "]";
  }

}  // namespace

double TunableKernel::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
  return 0.0;
}

double TunableKernel::estimatedTflops(const VulkanTuneParams& cfg, double kernelsPerSecond) const {
  if(kernelsPerSecond <= 0.0)
    return 0.0;
  double flops = estimatedFlopsPerDispatch(cfg);
  if(flops <= 0.0)
    return 0.0;
  return flops * kernelsPerSecond / 1e12;
}

// ============================================================================
// ComputeKernel lifecycle
// ============================================================================

// ComputeKernel::destroy

void ComputeKernel::destroy(VkDevice dev) {
  if(pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(dev, pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
  }
  if(pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
  }
  if(dsLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(dev, dsLayout, nullptr);
    dsLayout = VK_NULL_HANDLE;
  }
  if(shaderModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(dev, shaderModule, nullptr);
    shaderModule = VK_NULL_HANDLE;
  }
}

// ComputeKernel::build

ComputeKernel ComputeKernel::build(
  VkDevice device,
  const uint32_t* spirvData,
  size_t spirvWords,
  uint32_t numBindings,
  uint32_t pushConstantBytes,
  const vector<VkSpecializationMapEntry>& specEntries,
  const vector<uint32_t>& specData,
  VkPipelineCache pipelineCache,
  bool requireFullSubgroups,
  uint32_t requiredSubgroupSize,
  std::string_view pipelineStatsLabel) {
  ComputeKernel entry;
  entry.numBindings = numBindings;
  entry.shaderModule = VulkanHelpers::createShaderModule(device, spirvData, spirvWords);

  vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
  for(uint32_t i = 0; i < numBindings; i++) {
    bindings[i] = {};
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dsCI = {};
  dsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  dsCI.bindingCount = numBindings;
  dsCI.pBindings = numBindings > 0 ? bindings.data() : nullptr;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dsCI, nullptr, &entry.dsLayout));

  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = pushConstantBytes;

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1;
  plCI.pSetLayouts = &entry.dsLayout;
  plCI.pushConstantRangeCount = (pushConstantBytes > 0) ? 1 : 0;
  plCI.pPushConstantRanges = (pushConstantBytes > 0) ? &pcRange : nullptr;
  VK_CHECK(vkCreatePipelineLayout(device, &plCI, nullptr, &entry.pipelineLayout));

  VkSpecializationInfo specInfo = {};
  specInfo.mapEntryCount = (uint32_t)specEntries.size();
  specInfo.pMapEntries = specEntries.empty() ? nullptr : specEntries.data();
  specInfo.dataSize = specData.size() * sizeof(uint32_t);
  specInfo.pData = specData.empty() ? nullptr : specData.data();

  VkComputePipelineCreateInfo cpCI = {};
  cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpCI.stage.module = entry.shaderModule;
  cpCI.stage.pName = "main";
  cpCI.stage.pSpecializationInfo = specEntries.empty() ? nullptr : &specInfo;
  cpCI.stage.pNext = nullptr;
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroup = {};
  if(requireFullSubgroups || requiredSubgroupSize > 0) {
    testAssert(requiredSubgroupSize > 0);
    if(requireFullSubgroups)
      cpCI.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    reqSubgroup.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    reqSubgroup.requiredSubgroupSize = requiredSubgroupSize;
    cpCI.stage.pNext = &reqSubgroup;
  }
#else
  (void)requireFullSubgroups;
  (void)requiredSubgroupSize;
#endif
  cpCI.layout = entry.pipelineLayout;
#if defined(VK_KHR_pipeline_executable_properties)
  if(!pipelineStatsLabel.empty() && VulkanHelpers::pipelineExecutableStatsEnabled(device))
    cpCI.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
#else
  (void)pipelineStatsLabel;
#endif
  VK_CHECK(vkCreateComputePipelines(device, pipelineCache, 1, &cpCI, nullptr, &entry.pipeline));
  if(!pipelineStatsLabel.empty())
    VulkanHelpers::dumpPipelineExecutableStats(device, entry.pipeline, pipelineStatsLabel);

  return entry;
}

// Dispatch helper
//
// Dispatch a compute pipeline using bindingMap to expand a small
// "logical buffers" list (e.g. {input, output, scale, bias, mask}) into the
// full binding list the shader expects.
//
// Why this lives at the pipeline level
// Most shaders ship a single source compiled twice — once with USE_FP16_STORAGE=0
// and once with USE_FP16_STORAGE=1 — picked via specialization constants at
// pipeline creation. The two variants reference different SPIR-V binding
// numbers because GLSL pins each storage-buffer variable to a literal binding:
// e.g. scale_bias_mask_act has the FP32 declarations at bindings 0..4 and the
// FP16 declarations at bindings 5..9, all coexisting in the SPIR-V. After the
// spec-constant fold, only one half is "statically used" by the compiled
// pipeline, but the host's descriptor-set layout (built from numBindings, see
// ComputeKernel::build) declares every binding the layout needs and the host
// must bind a valid VkBuffer at each one — Vulkan 1.1 has no support for
// leaving bindings null without VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT.
//
// So bindingMap is a per-pipeline routing rule: bindingMap[i] is the
// index into the call site's logical buffers list to bind at descriptor
// binding i.

namespace {

  VkWriteDescriptorSet makeWriteDesc(VkDescriptorSet set, uint32_t binding, const VkDescriptorBufferInfo* bufInfo) {
    VkWriteDescriptorSet wd = {};
    wd.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wd.dstSet = set;
    wd.dstBinding = binding;
    wd.descriptorCount = 1;
    wd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wd.pBufferInfo = bufInfo;
    return wd;
  }

  // Check that a proposed workgroup (x,y,z) fits BOTH the total-invocations
  // limit and the per-dimension size limits reported by the device.
  bool workGroupFits(uint32_t lsx, uint32_t lsy, uint32_t lsz, const VkPhysicalDeviceLimits& lim) {
    if((uint64_t)lsx * lsy * lsz > lim.maxComputeWorkGroupInvocations)
      return false;
    if(lsx > lim.maxComputeWorkGroupSize[0])
      return false;
    if(lsy > lim.maxComputeWorkGroupSize[1])
      return false;
    if(lsz > lim.maxComputeWorkGroupSize[2])
      return false;
    return true;
  }

}  // namespace

// ============================================================================
// VulkanKernels spec helpers
// ============================================================================

namespace {

  std::vector<VkSpecializationMapEntry> makeSpecMap(int n) {
    std::vector<VkSpecializationMapEntry> entries(n);
    for(int i = 0; i < n; i++) {
      entries[i].constantID = i;
      entries[i].offset = i * sizeof(uint32_t);
      entries[i].size = sizeof(uint32_t);
    }
    return entries;
  }

  std::vector<VkSpecializationMapEntry> makeSpecMap(std::initializer_list<uint32_t> constantIDs) {
    std::vector<VkSpecializationMapEntry> entries;
    entries.reserve(constantIDs.size());
    uint32_t offset = 0;
    for(uint32_t constantID: constantIDs) {
      entries.push_back({constantID, offset, sizeof(uint32_t)});
      offset += sizeof(uint32_t);
    }
    return entries;
  }

  std::vector<uint32_t> makeSpecData(std::initializer_list<uint32_t> vals) {
    return std::vector<uint32_t>(vals);
  }

}  // namespace

void ComputeKernel::dispatch(
  const CmdCtx& ctx,
  std::initializer_list<VulkanBuffer*> logicalBufs,
  const void* pushConstantData,
  uint32_t pushConstantSize,
  uint32_t gx,
  uint32_t gy,
  uint32_t gz) const {
  testAssert(!bindingMap.empty());

  VkCommandBuffer cmd = ctx.cmd;

  std::array<VulkanBuffer*, 16> logical{};
  testAssert(logicalBufs.size() <= logical.size());
  size_t logicalCount = 0;
  for(VulkanBuffer* b: logicalBufs)
    logical[logicalCount++] = b;

  const size_t numBindings_ = bindingMap.size();
  vector<VkDescriptorBufferInfo> bufInfos(numBindings_);
  vector<VkWriteDescriptorSet> writes(numBindings_);
  for(size_t i = 0; i < numBindings_; i++) {
    uint8_t idx = bindingMap[i];
    testAssert(idx < logicalCount);
    VulkanBuffer* b = logical[idx];
    bufInfos[i] = {b->buffer, 0, b->size};
    writes[i] = makeWriteDesc(VK_NULL_HANDLE, (uint32_t)i, &bufInfos[i]);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  if(ctx.pushDescFn && numBindings_ > 0) {
    ctx.pushDescFn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, (uint32_t)numBindings_, writes.data());
  }
  if(pushConstantSize > 0) {
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstantData);
  }

  uint32_t pairIdx = UINT32_MAX;
  if(ctx.profiler != nullptr && ctx.profiler->active() && !debugName.empty())
    pairIdx = ctx.profiler->recordStart(cmd, debugName, debugFp16);
  vkCmdDispatch(cmd, gx, gy, gz);
  if(pairIdx != UINT32_MAX)
    ctx.profiler->recordEnd(cmd, pairIdx);
}

// Shared build() bodies for the AccF32/AccF16 accumulator twins. Each pair
// differs only in the tune-param field prefix, shader bytes, debug name, and
// (for the coopmat families) the config-support check; everything else is
// byte-identical. Public build() members below are 2-line forwarders.
namespace {

  // WinogradGemmDot2 / WinogradGemmDot2AccF16.
  template<bool AccF16>
  ComputeKernel buildWinogradGemmDot2Twin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    const int32_t workgroupSize = AccF16 ? cfg.dot2AccF16WorkgroupSize : cfg.dot2WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.dot2AccF16BM : cfg.dot2BM;
    const int32_t bn = AccF16 ? cfg.dot2AccF16BN : cfg.dot2BN;
    const int32_t sgm = AccF16 ? cfg.dot2AccF16SGM : cfg.dot2SGM;
    const int32_t sgn = AccF16 ? cfg.dot2AccF16SGN : cfg.dot2SGN;
    const int32_t sgmIter = AccF16 ? cfg.dot2AccF16SGMIter : cfg.dot2SGMIter;
    const int32_t tm = AccF16 ? cfg.dot2AccF16TM : cfg.dot2TM;
    const int32_t tn = AccF16 ? cfg.dot2AccF16TN : cfg.dot2TN;
    const int32_t subgroupSize = AccF16 ? cfg.dot2AccF16SubgroupSize : cfg.dot2SubgroupSize;
    testAssert(
      VulkanKernels::WinogradGemmDot2::isConfigSupported(workgroupSize, bm, bn, sgm, sgn, sgmIter, tm, tn, subgroupSize));
    testAssert(requiredSubgroupSize == 0 || (uint32_t)subgroupSize == requiredSubgroupSize);
    testAssert(k > 0);
    testAssert(k % VulkanKernels::DOT2_BK == 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(10);
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize, (uint32_t)bm, (uint32_t)bn, (uint32_t)sgm, (uint32_t)sgn, (uint32_t)sgmIter,
       (uint32_t)tm, (uint32_t)tn, (uint32_t)subgroupSize, (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? winograd_gemm_dot2_accf16 : winograd_gemm_dot2_accf32,
      AccF16 ? winograd_gemm_dot2_accf16_size : winograd_gemm_dot2_accf32_size,
      3,
      sizeof(VulkanKernels::WinogradGemmDot2::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ requiredSubgroupSize > 0,
      requiredSubgroupSize);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = AccF16 ? "WinogradGemmDot2AccF16" : "WinogradGemmDot2AccF32";
    kernel.debugFp16 = true;
    return kernel;
  }

  // GemmStridedDot2Nhwc / GemmStridedDot2AccF16Nhwc.
  template<bool AccF16>
  ComputeKernel buildGemmStridedDot2NhwcTwin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t packedB,
    int32_t kAligned,
    int32_t k,
    bool addToOutput,
    uint32_t requiredSubgroupSize) {
    const int32_t workgroupSize = AccF16 ? cfg.nhwcStridedDot2AccF16WorkgroupSize : cfg.nhwcStridedDot2WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.nhwcStridedDot2AccF16BM : cfg.nhwcStridedDot2BM;
    const int32_t bn = AccF16 ? cfg.nhwcStridedDot2AccF16BN : cfg.nhwcStridedDot2BN;
    const int32_t sgm = AccF16 ? cfg.nhwcStridedDot2AccF16SGM : cfg.nhwcStridedDot2SGM;
    const int32_t sgn = AccF16 ? cfg.nhwcStridedDot2AccF16SGN : cfg.nhwcStridedDot2SGN;
    const int32_t sgmIter = AccF16 ? cfg.nhwcStridedDot2AccF16SGMIter : cfg.nhwcStridedDot2SGMIter;
    const int32_t tm = AccF16 ? cfg.nhwcStridedDot2AccF16TM : cfg.nhwcStridedDot2TM;
    const int32_t tn = AccF16 ? cfg.nhwcStridedDot2AccF16TN : cfg.nhwcStridedDot2TN;
    const int32_t subgroupSize = AccF16 ? cfg.nhwcStridedDot2AccF16SubgroupSize : cfg.nhwcStridedDot2SubgroupSize;
    testAssert(VulkanKernels::GemmStridedDot2Nhwc::isConfigSupported(
      workgroupSize, bm, bn, sgm, sgn, sgmIter, tm, tn, subgroupSize));
    testAssert(requiredSubgroupSize == 0 || (uint32_t)subgroupSize == requiredSubgroupSize);
    testAssert(k > 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(14);
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)sgm,
       (uint32_t)sgn,
       (uint32_t)sgmIter,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)subgroupSize,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? gemm_strided_dot2_accf16_nhwc : gemm_strided_dot2_accf32_nhwc,
      AccF16 ? gemm_strided_dot2_accf16_nhwc_size : gemm_strided_dot2_accf32_nhwc_size,
      3,
      sizeof(VulkanKernels::GemmStridedDot2Nhwc::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ requiredSubgroupSize > 0,
      requiredSubgroupSize);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = AccF16 ? "GemmStridedDot2AccF16Nhwc" : "GemmStridedDot2AccF32Nhwc";
    kernel.debugFp16 = true;
    return kernel;
  }

  // WinogradGemmCoopmat1 / WinogradGemmCoopmat1AccF16.
  template<bool AccF16>
  ComputeKernel buildWinogradGemmCoopmat1Twin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    const int32_t workgroupSize = AccF16 ? cfg.coopmat1AccF16WorkgroupSize : cfg.coopmat1WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.coopmat1AccF16BM : cfg.coopmat1BM;
    const int32_t bn = AccF16 ? cfg.coopmat1AccF16BN : cfg.coopmat1BN;
    const int32_t bk = AccF16 ? cfg.coopmat1AccF16BK : cfg.coopmat1BK;
    const int32_t sgm = AccF16 ? cfg.coopmat1AccF16SGM : cfg.coopmat1SGM;
    const int32_t sgn = AccF16 ? cfg.coopmat1AccF16SGN : cfg.coopmat1SGN;
    const int32_t tm = AccF16 ? cfg.coopmat1AccF16TM : cfg.coopmat1TM;
    const int32_t tn = AccF16 ? cfg.coopmat1AccF16TN : cfg.coopmat1TN;
    const int32_t tk = AccF16 ? cfg.coopmat1AccF16TK : cfg.coopmat1TK;
    const int32_t subgroupSize = AccF16 ? cfg.coopmat1AccF16SubgroupSize : cfg.coopmat1SubgroupSize;
    if constexpr(AccF16)
      testAssert(VulkanKernels::WinogradGemmCoopmat1AccF16::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    else
      testAssert(VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    // SUBGROUP_SIZE (the shader's per-subgroup lane count spec constant) must equal the
    // pipeline's required subgroup size; coopmat fragments are subgroup-scoped,
    // so a mismatch would map lanes to the wrong fragment elements.
    testAssert((uint32_t)subgroupSize == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap(11);
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)sgm,
       (uint32_t)sgn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)subgroupSize,
       (uint32_t)k});
    const char* debugName = AccF16 ? "WinogradGemmCoopmat1AccF16" : "WinogradGemmCoopmat1AccF32";
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmatPipelineStatsLabel(debugName, workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? winograd_gemm_coopmat1_accf16 : winograd_gemm_coopmat1_accf32,
      AccF16 ? winograd_gemm_coopmat1_accf16_size : winograd_gemm_coopmat1_accf32_size,
      3,
      sizeof(VulkanKernels::WinogradGemmCoopmat1::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = debugName;
    kernel.debugFp16 = true;
    return kernel;
#else
    // Coopmat shaders were not built by this toolchain; supportsCoopmat1F16 is
    // forced false at detection, so this path is never selected at runtime.
    (void)device;
    (void)cache;
    return failUncompiledCoopmat();
#endif
  }

  // GemmStridedCoopmat1Nhwc / GemmStridedCoopmat1AccF16Nhwc.
  template<bool AccF16>
  ComputeKernel buildGemmStridedCoopmat1NhwcTwin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t k,
    uint32_t requiredSubgroupSize,
    bool addToOutput) {
    const int32_t workgroupSize =
      AccF16 ? cfg.nhwcStridedCoopmat1AccF16WorkgroupSize : cfg.nhwcStridedCoopmat1WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.nhwcStridedCoopmat1AccF16BM : cfg.nhwcStridedCoopmat1BM;
    const int32_t bn = AccF16 ? cfg.nhwcStridedCoopmat1AccF16BN : cfg.nhwcStridedCoopmat1BN;
    const int32_t bk = AccF16 ? cfg.nhwcStridedCoopmat1AccF16BK : cfg.nhwcStridedCoopmat1BK;
    const int32_t sgm = AccF16 ? cfg.nhwcStridedCoopmat1AccF16SGM : cfg.nhwcStridedCoopmat1SGM;
    const int32_t sgn = AccF16 ? cfg.nhwcStridedCoopmat1AccF16SGN : cfg.nhwcStridedCoopmat1SGN;
    const int32_t tm = AccF16 ? cfg.nhwcStridedCoopmat1AccF16TM : cfg.nhwcStridedCoopmat1TM;
    const int32_t tn = AccF16 ? cfg.nhwcStridedCoopmat1AccF16TN : cfg.nhwcStridedCoopmat1TN;
    const int32_t tk = AccF16 ? cfg.nhwcStridedCoopmat1AccF16TK : cfg.nhwcStridedCoopmat1TK;
    const int32_t subgroupSize = AccF16 ? cfg.nhwcStridedCoopmat1AccF16SubgroupSize : cfg.nhwcStridedCoopmat1SubgroupSize;
    if constexpr(AccF16)
      testAssert(VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    else
      testAssert(VulkanKernels::GemmStridedCoopmat1Nhwc::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    testAssert(k > 0 && (uint32_t)subgroupSize == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14});
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)sgm,
       (uint32_t)sgn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)subgroupSize,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)k});
    const char* debugName = AccF16 ? "GemmStridedCoopmat1AccF16Nhwc" : "GemmStridedCoopmat1AccF32Nhwc";
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmatPipelineStatsLabel(debugName, workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? gemm_strided_coopmat1_accf16_nhwc : gemm_strided_coopmat1_accf32_nhwc,
      AccF16 ? gemm_strided_coopmat1_accf16_nhwc_size : gemm_strided_coopmat1_accf32_nhwc_size,
      3,
      sizeof(VulkanKernels::GemmStridedCoopmat1Nhwc::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = debugName;
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)addToOutput;
    return failUncompiledCoopmat();
#endif
  }

  // WinogradGemmCoopmat2 / WinogradGemmCoopmat2AccF16.
  template<bool AccF16>
  ComputeKernel buildWinogradGemmCoopmat2Twin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k) {
    const int32_t workgroupSize = AccF16 ? cfg.coopmat2AccF16WorkgroupSize : cfg.coopmat2WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.coopmat2AccF16BM : cfg.coopmat2BM;
    const int32_t bn = AccF16 ? cfg.coopmat2AccF16BN : cfg.coopmat2BN;
    const int32_t bk = AccF16 ? cfg.coopmat2AccF16BK : cfg.coopmat2BK;
    static_assert(sizeof(VulkanKernels::WinogradGemmCoopmat2::PC) <= 128, "coopmat2 push constants exceed Vulkan minimum");
    if constexpr(AccF16)
      testAssert(VulkanKernels::WinogradGemmCoopmat2AccF16::isConfigSupported(workgroupSize, bm, bn, bk));
    else
      testAssert(VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(workgroupSize, bm, bn, bk));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap(5);
    auto specDat = makeSpecData({(uint32_t)workgroupSize, (uint32_t)bm, (uint32_t)bn, (uint32_t)bk, (uint32_t)k});
    const char* debugName = AccF16 ? "WinogradGemmCoopmat2AccF16" : "WinogradGemmCoopmat2";
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmat2PipelineStatsLabel(debugName, workgroupSize, bm, bn, bk);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? winograd_gemm_coopmat2_accf16 : winograd_gemm_coopmat2_accf32,
      AccF16 ? winograd_gemm_coopmat2_accf16_size : winograd_gemm_coopmat2_accf32_size,
      3,
      sizeof(VulkanKernels::WinogradGemmCoopmat2::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ false,
      /*requiredSubgroupSize*/ 0,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = debugName;
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    return failUncompiledCoopmat();
#endif
  }

  // GemmStridedCoopmat2Nhwc / GemmStridedCoopmat2AccF16Nhwc.
  template<bool AccF16>
  ComputeKernel buildGemmStridedCoopmat2NhwcTwin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t kAligned,
    int32_t k,
    bool addToOutput) {
    const int32_t workgroupSize =
      AccF16 ? cfg.nhwcStridedCoopmat2AccF16WorkgroupSize : cfg.nhwcStridedCoopmat2WorkgroupSize;
    const int32_t bm = AccF16 ? cfg.nhwcStridedCoopmat2AccF16BM : cfg.nhwcStridedCoopmat2BM;
    const int32_t bn = AccF16 ? cfg.nhwcStridedCoopmat2AccF16BN : cfg.nhwcStridedCoopmat2BN;
    const int32_t bk = AccF16 ? cfg.nhwcStridedCoopmat2AccF16BK : cfg.nhwcStridedCoopmat2BK;
    if constexpr(AccF16)
      testAssert(VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::isConfigSupported(workgroupSize, bm, bn, bk));
    else
      testAssert(VulkanKernels::GemmStridedCoopmat2Nhwc::isConfigSupported(workgroupSize, bm, bn, bk));
    testAssert(k > 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap({0, 1, 2, 3, 4, 5, 7, 8});
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)kAligned,
       (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? gemm_strided_coopmat2_accf16_nhwc : gemm_strided_coopmat2_accf32_nhwc,
      AccF16 ? gemm_strided_coopmat2_accf16_nhwc_size : gemm_strided_coopmat2_accf32_nhwc_size,
      3,
      sizeof(VulkanKernels::GemmStridedCoopmat2Nhwc::PC),
      specMap,
      specDat,
      cache,
      false,
      0);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = AccF16 ? "GemmStridedCoopmat2AccF16Nhwc" : "GemmStridedCoopmat2AccF32Nhwc";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)kAligned;
    (void)addToOutput;
    return failUncompiledCoopmat();
#endif
  }

  // Conv3x3ImplicitGemmCoopmat2AccF32/F16NhwcVec8.
  template<bool AccF16>
  ComputeKernel buildConv3x3ImplicitGemmCoopmat2Vec8Twin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t packedK,
    bool addToOutput) {
    const bool is5x5 = convSize == 5;
    const int32_t workgroupSize =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat2AccF16WorkgroupSize : cfg.conv3x3NhwcCoopmat2AccF16WorkgroupSize)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat2WorkgroupSize : cfg.conv3x3NhwcCoopmat2WorkgroupSize);
    const int32_t bm =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat2AccF16BM : cfg.conv3x3NhwcCoopmat2AccF16BM)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat2BM : cfg.conv3x3NhwcCoopmat2BM);
    const int32_t bn =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat2AccF16BN : cfg.conv3x3NhwcCoopmat2AccF16BN)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat2BN : cfg.conv3x3NhwcCoopmat2BN);
    const int32_t bk =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat2AccF16BK : cfg.conv3x3NhwcCoopmat2AccF16BK)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat2BK : cfg.conv3x3NhwcCoopmat2BK);
    if constexpr(AccF16)
      testAssert(VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(workgroupSize, bm, bn, bk));
    else
      testAssert(VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(workgroupSize, bm, bn, bk));
    testAssert((convSize == 3 || convSize == 5) && packedK > 0);
    const int32_t convTaps = convSize * convSize;
    testAssert(packedK % convTaps == 0 && (packedK / convTaps) % 8 == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    // Slot 17 marks whether a BK slab stays inside one convolution tap.
    auto specMap = makeSpecMap({0, 1, 2, 3, 11, 14, 17});
    const uint32_t slabInsideTap = (packedK / convTaps) % bk == 0 ? 1u : 0u;
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       addToOutput ? 1u : 0u,
       (uint32_t)packedK,
       slabInsideTap});
    // 3x3 deliberately uses the shader default. 5x5 gets the one additional
    // specialization entry that selects its independently compiled variant.
    if(convSize == 5) {
      specMap.push_back({18u, (uint32_t)(specDat.size() * sizeof(uint32_t)), sizeof(uint32_t)});
      specDat.push_back(5u);
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? conv_implicit_gemm_coopmat2_accf16_nhwc : conv_implicit_gemm_coopmat2_accf32_nhwc,
      AccF16 ? conv_implicit_gemm_coopmat2_accf16_nhwc_size : conv_implicit_gemm_coopmat2_accf32_nhwc_size,
      6,
      sizeof(VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::PC),
      specMap,
      specDat,
      cache,
      false,
      0);
    kernel.bindingMap = {0, 1, 2, 0, 0, 0};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)packedK};
    kernel.debugName =
      AccF16 ? (convSize == 5 ? "Conv5x5ImplicitGemmCoopmat2AccF16NhwcHalf8" : "Conv3x3ImplicitGemmCoopmat2AccF16NhwcHalf8")
             : (convSize == 5 ? "Conv5x5ImplicitGemmCoopmat2AccF32NhwcHalf8" : "Conv3x3ImplicitGemmCoopmat2AccF32NhwcHalf8");
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)convSize;
    (void)addToOutput;
    return failUncompiledCoopmat();
#endif
  }

  // Conv3x3ImplicitGemmCoopmat1AccF32/F16NhwcVec8.
  template<bool AccF16>
  ComputeKernel buildConv3x3ImplicitGemmCoopmat1Vec8Twin(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t k,
    bool addToOutput,
    bool slabInsideTap,
    uint32_t requiredSubgroupSize) {
    const bool is5x5 = convSize == 5;
    const int32_t workgroupSize =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16WorkgroupSize : cfg.conv3x3NhwcCoopmat1AccF16WorkgroupSize)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1WorkgroupSize : cfg.conv3x3NhwcCoopmat1WorkgroupSize);
    const int32_t bm =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BM : cfg.conv3x3NhwcCoopmat1AccF16BM)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1BM : cfg.conv3x3NhwcCoopmat1BM);
    const int32_t bn =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BN : cfg.conv3x3NhwcCoopmat1AccF16BN)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1BN : cfg.conv3x3NhwcCoopmat1BN);
    const int32_t bk =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BK : cfg.conv3x3NhwcCoopmat1AccF16BK)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1BK : cfg.conv3x3NhwcCoopmat1BK);
    const int32_t sgm =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16SGM : cfg.conv3x3NhwcCoopmat1AccF16SGM)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1SGM : cfg.conv3x3NhwcCoopmat1SGM);
    const int32_t sgn =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16SGN : cfg.conv3x3NhwcCoopmat1AccF16SGN)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1SGN : cfg.conv3x3NhwcCoopmat1SGN);
    const int32_t tm =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TM : cfg.conv3x3NhwcCoopmat1AccF16TM)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1TM : cfg.conv3x3NhwcCoopmat1TM);
    const int32_t tn =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TN : cfg.conv3x3NhwcCoopmat1AccF16TN)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1TN : cfg.conv3x3NhwcCoopmat1TN);
    const int32_t tk =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TK : cfg.conv3x3NhwcCoopmat1AccF16TK)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1TK : cfg.conv3x3NhwcCoopmat1TK);
    const int32_t subgroupSize =
      AccF16 ? (is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16SubgroupSize : cfg.conv3x3NhwcCoopmat1AccF16SubgroupSize)
             : (is5x5 ? cfg.conv5x5NhwcCoopmat1SubgroupSize : cfg.conv3x3NhwcCoopmat1SubgroupSize);
    static_assert(
      sizeof(VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::PC) <= 128,
      "implicit convolution vec8 push constants exceed Vulkan minimum");
    if constexpr(AccF16)
      testAssert(VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    else
      testAssert(VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
        workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize));
    testAssert((convSize == 3 || convSize == 5) && k > 0 && (uint32_t)subgroupSize == requiredSubgroupSize);
    const int32_t convTaps = convSize * convSize;
    testAssert(k % convTaps == 0 && (k / convTaps) % 8 == 0);
    testAssert(!slabInsideTap || (k / convTaps) % bk == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 17});
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)sgm,
       (uint32_t)sgn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)subgroupSize,
       addToOutput ? 1u : 0u,
       (uint32_t)k,
       slabInsideTap ? 1u : 0u});
    if(convSize == 5) {
      specMap.push_back({18u, (uint32_t)(specDat.size() * sizeof(uint32_t)), sizeof(uint32_t)});
      specDat.push_back(5u);
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      AccF16 ? conv_implicit_gemm_coopmat1_accf16_nhwc : conv_implicit_gemm_coopmat1_accf32_nhwc,
      AccF16 ? conv_implicit_gemm_coopmat1_accf16_nhwc_size : conv_implicit_gemm_coopmat1_accf32_nhwc_size,
      6,
      sizeof(VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize);
    // scale/bias/mask are unused but the shader deliberately retains the common
    // six-binding layout; alias them to the input buffer.
    kernel.bindingMap = {0, 1, 2, 0, 0, 0};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName =
      AccF16 ? (convSize == 5 ? "Conv5x5ImplicitGemmCoopmat1AccF16NhwcVec8" : "Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8")
             : (convSize == 5 ? "Conv5x5ImplicitGemmCoopmat1AccF32NhwcVec8" : "Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8");
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)convSize;
    (void)addToOutput;
    (void)slabInsideTap;
    testAssert(false && "Conv3x3 vec8 build called without compiled coopmat shaders");
    return ComputeKernel();
#endif
  }

  // Shared dispatch for the three Winograd accelerated GEMM variants, which
  // differ only in the K-alignment assertion (DOT2_BK vs 8).
  template<int K_MOD, typename PC>
  void dispatchWinogradGemmTile(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    testAssert(k > 0);
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && k % K_MOD == 0);
    const auto& tile = requireLaunchProfile<LaunchProfile::GemmDot2Tile>(kernel);
    testAssert(tile.bm > 0 && tile.bn > 0);
    testAssert(pc.M % tile.bm == 0 && pc.N % tile.bn == 0);
    testAssert(tile.kSpec == 0 || k == (int)tile.kSpec);
    kernel.dispatch(
      ctx, {A, B, C}, &pc, sizeof(PC), divUpU32(pc.M, tile.bm), divUpU32(pc.N, tile.bn), (uint32_t)numBatches);
  }

  // Shared dispatch for the three strided GEMM variants, which differ only in
  // the M/N-alignment assertion (coopmat2 accepts 4, the rest require 8).
  template<int M_MOD, typename PC>
  void dispatchGemmStridedTile(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    testAssert(input != nullptr && output != nullptr && input != output && input->buffer != output->buffer);
    testAssert(k > 0);
    testAssert(pc.M % M_MOD == 0 && pc.N % M_MOD == 0 && pc.N_real <= pc.N);
    const auto& tile = requireLaunchProfile<LaunchProfile::GemmDot2Tile>(kernel);
    testAssert(tile.bm > 0 && tile.bn > 0);
    testAssert(tile.kSpec == 0 || k == (int)tile.kSpec);
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.M, tile.bm),
      divUpU32(pc.N, tile.bn),
      (uint32_t)numBatches);
  }

}  // namespace

// VulkanKernels — per-kernel build()/dispatch() bodies
// ============================================================================
// VulkanKernels implementations
// ============================================================================

namespace VulkanKernels {

  // -------- ScaleBiasMaskActNhwc --------

  ComputeKernel ScaleBiasMaskActNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int activation,
    bool useVec4,
    bool hasDynamicBias) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({(uint32_t)activation, fp16 ? 1u : 0u, useVec4 ? 1u : 0u, hasDynamicBias ? 1u : 0u});
    // Binding 5 is always the optional dynamic bias, before all FP16/vector
    // aliases, so every specialization has a compact contiguous layout.
    const uint32_t numBindings = useVec4 ? (fp16 ? 19u : 15u) : (fp16 ? 11u : 6u);
    ComputeKernel kernel = ComputeKernel::build(
      device,
      scale_bias_mask_act_nhwc,
      scale_bias_mask_act_nhwc_size,
      numBindings,
      sizeof(PC),
      specMap,
      specDat,
      cache);
    if(useVec4) {
      kernel.bindingMap = {
        0,
        1,
        2,
        3,
        4,
        hasDynamicBias ? uint8_t{5} : uint8_t{0},
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
        3,
      };
      if(fp16)
        kernel.bindingMap.insert(kernel.bindingMap.end(), {0, 1, 2, 3});
    } else {
      kernel.bindingMap = {0, 1, 2, 3, 4, hasDynamicBias ? uint8_t{5} : uint8_t{0}};
      if(fp16)
        kernel.bindingMap.insert(kernel.bindingMap.end(), {0, 1, 2, 3, 4});
    }
    testAssert(kernel.bindingMap.size() == kernel.numBindings);
    kernel.debugName = useVec4
                         ? (hasDynamicBias ? "ScaleBiasMaskActNhwcVec4DynamicBias" : "ScaleBiasMaskActNhwcVec4")
                         : (hasDynamicBias ? "ScaleBiasMaskActNhwcTiledDynamicBias" : "ScaleBiasMaskActNhwcTiled");
    kernel.debugFp16 = fp16;
    kernel.localSizeX = LOCAL_SIZE_X;
    kernel.localSizeY = LOCAL_SIZE_Y;
    kernel.launch.data = LaunchProfile::NhwcChannelTile{useVec4 ? uint8_t{4} : uint8_t{1}};
    return kernel;
  }

  void ScaleBiasMaskActNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* scale,
    VulkanBuffer* bias,
    VulkanBuffer* mask,
    const PC& pc,
    VulkanBuffer* dynamicBias) {
    const auto& profile = requireLaunchProfile<LaunchProfile::NhwcChannelTile>(kernel);
    testAssert(profile.channelsPerInvocation == 1 || profile.channelsPerInvocation == 4);
    testAssert(profile.channelsPerInvocation == 1 || pc.numChannels % 4 == 0);
    const uint32_t channelsPerGroup = kernel.localSizeX * profile.channelsPerInvocation;
    const uint32_t groupsX = divUpU32((uint32_t)pc.numChannels, channelsPerGroup);
    const uint32_t groupsY = divUpU32((uint32_t)pc.paddedSpatialSize, kernel.localSizeY);
    if(dynamicBias != nullptr) {
      kernel.dispatch(
        ctx,
        {input, output, scale, bias, mask, dynamicBias},
        &pc,
        sizeof(PC),
        groupsX,
        groupsY,
        (uint32_t)pc.batchSize);
    } else {
      kernel.dispatch(
        ctx, {input, output, scale, bias, mask}, &pc, sizeof(PC), groupsX, groupsY, (uint32_t)pc.batchSize);
    }
  }

  // -------- ScaleBiasMaskActInplaceNhwc --------

  ComputeKernel ScaleBiasMaskActInplaceNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int activation,
    bool useVec4,
    bool hasDynamicBias) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({(uint32_t)activation, fp16 ? 1u : 0u, useVec4 ? 1u : 0u, hasDynamicBias ? 1u : 0u});
    const uint32_t numBindings = useVec4 ? (fp16 ? 15u : 12u) : (fp16 ? 9u : 5u);
    ComputeKernel kernel = ComputeKernel::build(
      device,
      scale_bias_mask_act_inplace_nhwc,
      scale_bias_mask_act_inplace_nhwc_size,
      numBindings,
      sizeof(PC),
      specMap,
      specDat,
      cache);
    if(useVec4) {
      kernel.bindingMap = {
        0,
        1,
        2,
        3,
        hasDynamicBias ? uint8_t{4} : uint8_t{0},
        0,
        1,
        2,
        3,
        0,
        1,
        2,
      };
      if(fp16)
        kernel.bindingMap.insert(kernel.bindingMap.end(), {0, 1, 2});
    } else {
      kernel.bindingMap = {0, 1, 2, 3, hasDynamicBias ? uint8_t{4} : uint8_t{0}};
      if(fp16)
        kernel.bindingMap.insert(kernel.bindingMap.end(), {0, 1, 2, 3});
    }
    testAssert(kernel.bindingMap.size() == kernel.numBindings);
    kernel.debugName =
      useVec4 ? (hasDynamicBias ? "ScaleBiasMaskActInplaceNhwcVec4DynamicBias" : "ScaleBiasMaskActInplaceNhwcVec4")
              : (hasDynamicBias ? "ScaleBiasMaskActInplaceNhwcTiledDynamicBias" : "ScaleBiasMaskActInplaceNhwcTiled");
    kernel.debugFp16 = fp16;
    kernel.localSizeX = LOCAL_SIZE_X;
    kernel.localSizeY = LOCAL_SIZE_Y;
    kernel.launch.data = LaunchProfile::NhwcChannelTile{useVec4 ? uint8_t{4} : uint8_t{1}};
    return kernel;
  }

  void ScaleBiasMaskActInplaceNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* tensor,
    VulkanBuffer* scale,
    VulkanBuffer* bias,
    VulkanBuffer* mask,
    const PC& pc,
    VulkanBuffer* dynamicBias) {
    const auto& profile = requireLaunchProfile<LaunchProfile::NhwcChannelTile>(kernel);
    const uint32_t channelsPerGroup = kernel.localSizeX * profile.channelsPerInvocation;
    const uint32_t groupsX = divUpU32((uint32_t)pc.numChannels, channelsPerGroup);
    const uint32_t groupsY = divUpU32((uint32_t)pc.paddedSpatialSize, kernel.localSizeY);
    if(dynamicBias != nullptr)
      kernel.dispatch(
        ctx, {tensor, scale, bias, mask, dynamicBias}, &pc, sizeof(PC), groupsX, groupsY, (uint32_t)pc.batchSize);
    else
      kernel.dispatch(ctx, {tensor, scale, bias, mask}, &pc, sizeof(PC), groupsX, groupsY, (uint32_t)pc.batchSize);
  }

  // -------- NchwToNhwc --------

  void LayoutTransformKernels::destroy(VkDevice device) {
    direct.destroy(device);
    tiledSmall.destroy(device);
    tiledLarge.destroy(device);
  }

  static ComputeKernel buildLayoutTransformTiled(VkDevice device, VkPipelineCache cache, bool fp16, uint32_t tileDim) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(3);
    auto specDat = makeSpecData({fp16 ? 1u : 0u, tileDim, tileDim});
    ComputeKernel kernel =
      ComputeKernel::build(device, nchw_to_nhwc_tiled, nchw_to_nhwc_tiled_size, 4, sizeof(NchwToNhwc::PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0, 1} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "NchwToNhwcTiled";
    kernel.debugFp16 = fp16;
    kernel.localSizeX = (uint16_t)tileDim;
    kernel.localSizeY = (uint16_t)LayoutTransformKernels::TILE_LOCAL_SIZE_Y;
    return kernel;
  }

  NchwToNhwc::Kernels
  NchwToNhwc::build(VkDevice device, VkPipelineCache cache, bool fp16, const VulkanTuneParams& tuneParams) {
    using namespace VulkanShaders;
    testAssert(
      LayoutTransform::isConfigSupported(
        tuneParams.nchwToNhwcSmallTile, tuneParams.nchwToNhwcLargeTile, tuneParams.nchwToNhwcTileCrossover));
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    Kernels kernels;
    kernels.direct = ComputeKernel::build(
      device, nchw_to_nhwc_smallchannel, nchw_to_nhwc_smallchannel_size, 6, sizeof(PC), specMap, specDat, cache);
    kernels.direct.bindingMap = std::vector<uint8_t>{0, 1, 0, 1, 1, 1};
    kernels.direct.debugName = "NchwToNhwcDirect";
    kernels.direct.debugFp16 = fp16;
    kernels.direct.localSizeX = (uint16_t)Kernels::DIRECT_LOCAL_SIZE_X;
    kernels.direct.localSizeY = 1;
    kernels.smallTileDim = (uint32_t)tuneParams.nchwToNhwcSmallTile;
    kernels.largeTileDim = (uint32_t)tuneParams.nchwToNhwcLargeTile;
    kernels.tileCrossover = (uint32_t)tuneParams.nchwToNhwcTileCrossover;
    kernels.tiledSmall = buildLayoutTransformTiled(device, cache, fp16, kernels.smallTileDim);
    kernels.tiledLarge = buildLayoutTransformTiled(device, cache, fp16, kernels.largeTileDim);
    return kernels;
  }

  void NchwToNhwc::dispatch(
    const CmdCtx& ctx,
    const Kernels& kernels,
    VulkanBuffer* input,
    VulkanBuffer* output,
    const PC& pc) {
    testAssert(input != nullptr && output != nullptr && input->buffer != output->buffer);
    testAssert(pc.inputChannels > 0 && pc.outputChannels >= pc.inputChannels);
    const bool packed = pc.inputChannels == pc.outputChannels && (pc.outputChannels == 2 || pc.outputChannels == 4);
    if((uint32_t)pc.outputChannels <= Kernels::DIRECT_MAX_CHANNELS) {
      uint32_t total;
      if(packed) {
        testAssert(pc.outputChannels != 2 || pc.paddedSpatialSize % 2 == 0);
        const uint32_t spatialPerInvocation = pc.outputChannels == 2 ? 2u : 1u;
        total = (uint32_t)pc.batchSize * (uint32_t)pc.paddedSpatialSize / spatialPerInvocation;
      } else {
        total = (uint32_t)pc.batchSize * (uint32_t)pc.outputChannels * (uint32_t)pc.paddedSpatialSize;
      }
      kernels.direct.dispatch(
        ctx, {input, output}, &pc, sizeof(PC), divUpU32(total, Kernels::DIRECT_LOCAL_SIZE_X), 1u, 1u);
      return;
    }

    const bool useSmallTile = (uint32_t)pc.outputChannels <= kernels.tileCrossover;
    const uint32_t tileDim = useSmallTile ? kernels.smallTileDim : kernels.largeTileDim;
    const ComputeKernel& tiled = useSmallTile ? kernels.tiledSmall : kernels.tiledLarge;
    tiled.dispatch(
      ctx,
      {input, output},
      &pc,
      sizeof(PC),
      divUpU32((uint32_t)pc.outputChannels, tileDim),
      divUpU32((uint32_t)pc.paddedSpatialSize, tileDim),
      (uint32_t)pc.batchSize);
  }

  bool LayoutTransform::isConfigSupported(int32_t smallTile, int32_t largeTile, int32_t tileCrossover) {
    const bool smallValid = smallTile == 8 || smallTile == 16 || smallTile == 32;
    const bool largeValid = largeTile == 16 || largeTile == 32;
    const bool crossoverValid = tileCrossover == 8 || tileCrossover == 16 || tileCrossover == 32;
    return smallValid && largeValid && smallTile <= largeTile && crossoverValid;
  }

  ArrayView<TunableParam> NchwToNhwcTuner::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_NCHW_TO_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }

  ArrayView<TunableParam> NchwToNhwcTuner::params() const {
    return sharedParams();
  }

  bool NchwToNhwcTuner::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const {
    if(!LayoutTransform::isConfigSupported(
         cfg.nchwToNhwcSmallTile, cfg.nchwToNhwcLargeTile, cfg.nchwToNhwcTileCrossover))
      return false;
    const uint32_t maxTile = (uint32_t)std::max(cfg.nchwToNhwcSmallTile, cfg.nchwToNhwcLargeTile);
    const uint64_t sharedBytes = (uint64_t)maxTile * (uint64_t)(maxTile + 1u) * sizeof(float);
    return maxTile <= limits.maxComputeWorkGroupSize[0] &&
           LayoutTransformKernels::TILE_LOCAL_SIZE_Y <= limits.maxComputeWorkGroupSize[1] &&
           maxTile * LayoutTransformKernels::TILE_LOCAL_SIZE_Y <= limits.maxComputeWorkGroupInvocations &&
           sharedBytes <= limits.maxComputeSharedMemorySize;
  }

  uint32_t NchwToNhwcTuner::workgroupThreads(const VulkanTuneParams& cfg) const {
    return (uint32_t)std::max(cfg.nchwToNhwcSmallTile, cfg.nchwToNhwcLargeTile) *
           LayoutTransformKernels::TILE_LOCAL_SIZE_Y;
  }

  // -------- WinogradTransformNhwc --------
  // Native NHWC Winograd input transform.

  ComputeKernel WinogradTransformNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    const VulkanTuneParams& cfg,
    int filterSize,
    int offset,
    int packedBM,
    int packedBK,
    int packedAPadWords) {
    // 5x5 is architecturally fixed to F(2,5); only 3x3 reads the tuned out-tile.
    const int outTile = (filterSize == 3) ? nhwcWinograd3x3OutTileFor(cfg) : 2;
    const int inTile = outTile + filterSize - 1;
    const int localSizeX = cfg.nhwcWinogradTransformLocalSizeX;
    const int localSizeY = cfg.nhwcWinogradTransformLocalSizeY;
    requireSupportedWinogradGeometry("WinogradTransformNhwc", filterSize, inTile, outTile);
    testAssert(isConfigSupported(localSizeX, localSizeY));
    testAssert(packedBM > 0 && packedBK > 0 && packedBK % 4 == 0 && packedAPadWords >= 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(14);
    auto specDat = makeSpecData(
      {(uint32_t)inTile,
       (uint32_t)inTile,
       (uint32_t)outTile,
       (uint32_t)outTile,
       (uint32_t)filterSize,
       (uint32_t)filterSize,
       (uint32_t)offset,
       (uint32_t)offset,
       fp16 ? 1u : 0u,
       (uint32_t)localSizeX,
       (uint32_t)localSizeY,
       (uint32_t)packedBM,
       (uint32_t)packedBK,
       (uint32_t)packedAPadWords});
    ComputeKernel kernel = ComputeKernel::build(
      device, winograd_transform_nhwc, winograd_transform_nhwc_size, 4, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 0, 1};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.debugName = "WinogradTransformNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradTransformNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    const PC& pc) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    testAssert(pc.numInChannelsPadded % 4 == 0);
    testAssert(pc.numTilesReal <= pc.numTilesPadded);
    kernel.dispatch(
      ctx,
      {input, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.numInChannelsPadded, kernel.localSizeX),
      divUpU32(pc.numTilesPadded, kernel.localSizeY),
      1u);
  }

  ArrayView<TunableParam> WinogradTransformNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_TRANSFORM_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> WinogradTransformNhwc::params() const {
    return sharedParams();
  }

  WinogradTransformNhwc::WinogradTransformNhwc(int batchSize, int inChannels, int nnXLen, int nnYLen)
    : problemBatchSize(batchSize), problemInChannels(inChannels), problemNnXLen(nnXLen), problemNnYLen(nnYLen) {}

  double WinogradTransformNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    int outTile = nhwcWinograd3x3OutTileFor(cfg);
    double flopsPerInvocation = (outTile == 2) ? 32.0 : 336.0;
    return winogradTransformInvocationCount(problemBatchSize, problemInChannels, problemNnXLen, problemNnYLen, cfg) *
           flopsPerInvocation;
  }

  bool WinogradTransformNhwc::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  bool WinogradTransformNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.nhwcWinogradTransformLocalSizeX, cfg.nhwcWinogradTransformLocalSizeY))
      return false;
    if(!workGroupFits(
         (uint32_t)cfg.nhwcWinogradTransformLocalSizeX, (uint32_t)cfg.nhwcWinogradTransformLocalSizeY, 1u, lim))
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- AddPointwise --------

  ComputeKernel AddPointwise::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, add_pointwise, add_pointwise_size, fp16 ? 4 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0, 1} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "AddPointwise";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void AddPointwise::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* acc,
    VulkanBuffer* value,
    const PC& pc) {
    kernel.dispatch(ctx, {acc, value}, &pc, sizeof(PC), divUpU32(pc.numElements, AddPointwise::LOCAL_SIZE_X), 1u, 1u);
  }

  // -------- AddChannelBiasesNhwc --------

  ComputeKernel AddChannelBiasesNhwc::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, add_channel_biases_nhwc, add_channel_biases_nhwc_size, fp16 ? 3 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "AddChannelBiasesNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void AddChannelBiasesNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* srcDst,
    VulkanBuffer* bias,
    const PC& pc) {
    kernel.dispatch(
      ctx,
      {srcDst, bias},
      &pc,
      sizeof(PC),
      divUpU32(pc.paddedSpatialSize, AddChannelBiasesNhwc::LOCAL_SIZE_X),
      (uint32_t)pc.batchChannelCount,
      1u);
  }

  // -------- AddCBiasActNC --------

  ComputeKernel AddCBiasActNC::build(VkDevice device, VkPipelineCache cache, int activation) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({(uint32_t)activation});
    ComputeKernel kernel =
      ComputeKernel::build(device, add_cbias_act_nc, add_cbias_act_nc_size, 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1};
    kernel.debugName = "AddCBiasActNC";
    kernel.debugFp16 = false;
    return kernel;
  }

  void AddCBiasActNC::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* data,
    VulkanBuffer* bias,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx,
      {data, bias},
      &pc,
      sizeof(PC),
      divUpU32((uint32_t)pc.numChannels, AddCBiasActNC::LOCAL_SIZE_X),
      (uint32_t)maxBatchSize,
      1u);
  }

  // -------- ExtractChannel0Nhwc --------

  ComputeKernel ExtractChannel0Nhwc::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, extract_channel0_nhwc, extract_channel0_nhwc_size, fp16 ? 4 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0, 1} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "ExtractChannel0Nhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void ExtractChannel0Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx,
      {input, mask},
      &pc,
      sizeof(PC),
      divUpU32(pc.paddedSpatialSize, ExtractChannel0Nhwc::LOCAL_SIZE_X),
      (uint32_t)maxBatchSize,
      1u);
  }

  // -------- SumMaskSpatial --------

  ComputeKernel SumMaskSpatial::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, sum_mask_spatial, sum_mask_spatial_size, fp16 ? 3 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "SumMaskSpatial";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void SumMaskSpatial::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(ctx, {mask, maskSum}, &pc, sizeof(PC), 1u, (uint32_t)maxBatchSize, 1u);
  }

  // -------- WinogradUntransformNhwc --------
  // Native NHWC Winograd output transform. ocVec (OC_VEC, spec constant 10)
  // channel-vectorizes the store.

  ComputeKernel WinogradUntransformNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    const VulkanTuneParams& cfg,
    int filterSize,
    bool addToOutput,
    int ocVec) {
    // 5x5 is architecturally fixed to F(2,5); only 3x3 reads the tuned out-tile.
    const int outTile = (filterSize == 3) ? nhwcWinograd3x3OutTileFor(cfg) : 2;
    const int inTile = outTile + filterSize - 1;
    const int localSizeX = cfg.nhwcWinogradUntransformLocalSizeX;
    const int localSizeY = cfg.nhwcWinogradUntransformLocalSizeY;
    requireSupportedWinogradGeometry("WinogradUntransformNhwc", filterSize, inTile, outTile);
    testAssert(isConfigSupported(localSizeX, localSizeY));
    testAssert(ocVec == 1 || ocVec == 4);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(11);
    auto specDat = makeSpecData(
      {(uint32_t)inTile,
       (uint32_t)inTile,
       (uint32_t)outTile,
       (uint32_t)outTile,
       (uint32_t)filterSize,
       (uint32_t)filterSize,
       fp16 ? 1u : 0u,
       (uint32_t)localSizeX,
       (uint32_t)localSizeY,
       addToOutput ? 1u : 0u,
       (uint32_t)ocVec});
    ComputeKernel kernel = ComputeKernel::build(
      device, winograd_untransform_nhwc, winograd_untransform_nhwc_size, 6, sizeof(PC), specMap, specDat, cache);
    // Bindings 0/1 (fp32 scalar) and 2/3 (fp16 scalar) alias the same VkBuffers
    // as 4/5 (fp32/fp16 vec4) -- see the OutputBufV4/OutputBufV4H comment in
    // the shader. The compiled pipeline only statically uses the scalar pair
    // matching `fp16` plus the vec4 pair matching `fp16`, but every binding
    // in the descriptor-set layout still needs a bound buffer (see the
    // bindingMap doc comment above), so all four output slots route to the
    // same logical output buffer (index 1).
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 0, 1, 1, 1};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.debugName = "WinogradUntransformNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradUntransformNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    const PC& pc,
    int maxBatchSize,
    int ocVec) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    testAssert(ocVec == 1 || ocVec == 4);
    testAssert(pc.numOutChannels % ocVec == 0);
    kernel.dispatch(
      ctx,
      {input, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.numTilesX, kernel.localSizeX),
      divUpU32(pc.numTilesY, kernel.localSizeY),
      (uint32_t)(maxBatchSize * (pc.numOutChannels / ocVec)));
  }

  bool WinogradUntransformNhwc::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  ArrayView<TunableParam> WinogradUntransformNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_UNTRANSFORM_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> WinogradUntransformNhwc::params() const {
    return sharedParams();
  }

  WinogradUntransformNhwc::WinogradUntransformNhwc(int batchSize, int outChannels, int nnXLen, int nnYLen)
    : problemBatchSize(batchSize), problemOutChannels(outChannels), problemNnXLen(nnXLen), problemNnYLen(nnYLen) {}

  double WinogradUntransformNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    if(problemBatchSize <= 0 || problemOutChannels <= 0 || problemNnXLen <= 0 || problemNnYLen <= 0)
      return 0.0;
    int outTile = nhwcWinograd3x3OutTileFor(cfg);
    int numTilesX = ceilDivInt(problemNnXLen, outTile);
    int numTilesY = ceilDivInt(problemNnYLen, outTile);
    double invocations = (double)problemBatchSize * (double)problemOutChannels * (double)numTilesX * (double)numTilesY;
    double flopsPerInvocation = (outTile == 2) ? 24.0 : 150.0;
    return invocations * flopsPerInvocation;
  }

  bool WinogradUntransformNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.nhwcWinogradUntransformLocalSizeX, cfg.nhwcWinogradUntransformLocalSizeY))
      return false;
    if(!workGroupFits(
         (uint32_t)cfg.nhwcWinogradUntransformLocalSizeX, (uint32_t)cfg.nhwcWinogradUntransformLocalSizeY, 1u, lim))
      return false;
    return true;
  }

  // -------- WinogradGemm --------

  ComputeKernel
  WinogradGemm::build(VkDevice device, VkPipelineCache cache, bool fp16, const VulkanTuneParams& cfg, int32_t k) {
    const int32_t tileM = cfg.winogradGemmM;
    const int32_t tileN = cfg.winogradGemmN;
    const int32_t tileK = cfg.winogradGemmK;
    const int32_t rn = cfg.winogradGemmRN;
    testAssert(isConfigSupported(tileM, tileN, tileK, rn));
    testAssert(k > 0);
    using namespace VulkanShaders;
    // Spec constants are the WORKGROUP dimensions (WG_X = tileM/4, WG_Y = tileN/rn),
    // not the macro tile dimensions. The shader derives TILE_M and TILE_N from them.
    // Vulkan requires the spec id used by local_size_*_id to BE the local-size value.
    uint32_t wgX = (uint32_t)(tileM / 4);
    uint32_t wgY = (uint32_t)(tileN / rn);
    auto specMap = makeSpecMap(6);
    auto specDat = makeSpecData({wgX, wgY, (uint32_t)tileK, (uint32_t)rn, fp16 ? 1u : 0u, (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device, winograd_gemm_tiled, winograd_gemm_tiled_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    // Logical buffers are always {A, B, C}; FP16 aliases them onto bindings 3..5.
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1, 2} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)wgX;
    kernel.localSizeY = (uint16_t)wgY;
    kernel.launch.data = LaunchProfile::GemmRegN{(uint16_t)rn, (uint32_t)k};
    kernel.debugName = "WinogradGemm";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradGemm::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    const auto& launch = requireLaunchProfile<LaunchProfile::GemmRegN>(kernel);
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0 && launch.regN > 0);
    testAssert(k > 0);
    testAssert(launch.kSpec == 0 || k == (int)launch.kSpec);
    // The tiled shader uses 8-wide A loads and vec4 B/C paths:
    //   M÷8 (A load stride uses M/8), N÷8 (host padded, B stride still N/4 vec4 units).
    // A misaligned dim truncates a stride and/or overruns on the boundary tile.
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && k % 4 == 0);
    // One workgroup per (TILE_M, TILE_N) macro tile of the output.
    int macroM = (int)kernel.localSizeX * 4;
    int macroN = (int)kernel.localSizeY * (int)launch.regN;
    kernel.dispatch(
      ctx, {A, B, C}, &pc, sizeof(PC), divUpU32(pc.M, macroM), divUpU32(pc.N, macroN), (uint32_t)numBatches);
  }

  ArrayView<TunableParam> WinogradGemm::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_TILED
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> WinogradGemm::params() const {
    return sharedParams();
  }

  WinogradGemm::WinogradGemm(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemm::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  bool WinogradGemm::isConfigSupported(int32_t tileM, int32_t tileN, int32_t tileK, int32_t rn) {
    // tileM÷8: 8-wide A loads. tileN÷8: aligned padded-N contract.
    // tileN%rn: each thread owns rn contiguous n-columns.
    // rn%4: tileB is vec4-typed so RN must be a multiple of 4.
    return tileM > 0 && tileN > 0 && tileK > 0 && rn > 0 && (tileM % 8 == 0) && (tileN % 8 == 0) && (tileN % rn == 0) &&
           (tileK % 4 == 0) && (rn % 4 == 0);
  }

  bool WinogradGemm::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.winogradGemmM, cfg.winogradGemmN, cfg.winogradGemmK, cfg.winogradGemmRN))
      return false;
    // tileA: scalar K-major [K][M+4], transposed while loading row-major packed A.
    // tileB: vec4 with +1 vec4 padding per row → (TILE_N4+1)*4 floats per K row.
    size_t tileN4 = (size_t)(cfg.winogradGemmN / 4);
    size_t tileABytes = (size_t)cfg.winogradGemmK * ((size_t)cfg.winogradGemmM + 4) * sizeof(float);
    size_t tileBBytes = (size_t)cfg.winogradGemmK * (tileN4 + 1) * 4 * sizeof(float);
    size_t sharedBytes = tileABytes + tileBBytes;
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    uint32_t lsx = (uint32_t)(cfg.winogradGemmM / 4);
    uint32_t lsy = (uint32_t)(cfg.winogradGemmN / cfg.winogradGemmRN);
    if(!workGroupFits(lsx, lsy, 1u, lim))
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- GemmStridedTiledNhwc --------

  ComputeKernel GemmStridedTiledNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    const VulkanTuneParams& cfg,
    int32_t k,
    bool addToOutput) {
    const int32_t localSizeX = cfg.gemmStridedTiledNhwcLocalSizeX;
    const int32_t localSizeY = cfg.gemmStridedTiledNhwcLocalSizeY;
    const int32_t tileK = cfg.gemmStridedTiledNhwcTileK;
    const int32_t rn = cfg.gemmStridedTiledNhwcRN;
    testAssert(GemmStridedTiledNhwc::isConfigSupported(localSizeX, localSizeY, tileK, rn));
    testAssert(k > 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(7);
    auto specDat = makeSpecData(
      {(uint32_t)localSizeX,
       (uint32_t)localSizeY,
       (uint32_t)tileK,
       (uint32_t)rn,
       addToOutput ? 1u : 0u,
       fp16 ? 1u : 0u,
       (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device, gemm_strided_tiled_nhwc, gemm_strided_tiled_nhwc_size, fp16 ? 8 : 3, sizeof(PC), specMap, specDat, cache);
    // Bindings 6/7 alias A and C with a vector element type so aligned NHWC
    // channel slabs load and store as vec4s. They exist only in FP16 storage.
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1, 2, 0, 2} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.launch.data = LaunchProfile::GemmRegN{(uint16_t)rn, (uint32_t)k};
    kernel.debugName = "GemmStridedTiledNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void GemmStridedTiledNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    testAssert(input != nullptr && output != nullptr && input != output && input->buffer != output->buffer);
    const auto& launch = requireLaunchProfile<LaunchProfile::GemmRegN>(kernel);
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0 && launch.regN > 0);
    testAssert(k > 0);
    testAssert(launch.kSpec == 0 || k == (int)launch.kSpec);
    // M is scalar-addressed in the NHWC A/C buffers. N remains packed
    // as vec4s in B, and the common ABI reports all batch strides in vec4s.
    testAssert(pc.M % 4 == 0 && pc.N % 8 == 0 && pc.N_real <= pc.N);
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.M, (int)kernel.localSizeX * 4),
      divUpU32(pc.N, (int)kernel.localSizeY * (int)launch.regN),
      (uint32_t)numBatches);
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  ArrayView<TunableParam> GemmStridedTiledNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_TILED
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> GemmStridedTiledNhwc::params() const {
    return sharedParams();
  }

  GemmStridedTiledNhwc::GemmStridedTiledNhwc(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedTiledNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  bool GemmStridedTiledNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    // Validate the MM=WG_X*4, NN=WG_Y*RN, shared-A vec4 tile, and TILE_K geometry.
    return isConfigSupported(
             cfg.gemmStridedTiledNhwcLocalSizeX,
             cfg.gemmStridedTiledNhwcLocalSizeY,
             cfg.gemmStridedTiledNhwcTileK,
             cfg.gemmStridedTiledNhwcRN) &&
           workGroupFits(
             (uint32_t)cfg.gemmStridedTiledNhwcLocalSizeX, (uint32_t)cfg.gemmStridedTiledNhwcLocalSizeY, 1u, lim) &&
           (size_t)cfg.gemmStridedTiledNhwcTileK *
               (((size_t)cfg.gemmStridedTiledNhwcLocalSizeX + 1) * 4 +
                ((size_t)cfg.gemmStridedTiledNhwcLocalSizeY * cfg.gemmStridedTiledNhwcRN + 1)) *
               sizeof(float) <=
             lim.maxComputeSharedMemorySize;
  }

  bool GemmStridedTiledNhwc::isConfigSupported(int32_t localSizeX, int32_t localSizeY, int32_t tileK, int32_t rn) {
    return localSizeX > 0 && (localSizeX % 2 == 0) && localSizeY > 0 && tileK > 0 && rn > 0 &&
           ((localSizeY * rn) % 4 == 0);
  }

  // -------- WinogradGemmDot2 --------

  ComputeKernel WinogradGemmDot2::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    return buildWinogradGemmDot2Twin<false>(device, cache, cfg, k, requiredSubgroupSize);
  }

  void WinogradGemmDot2::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchWinogradGemmTile<VulkanKernels::DOT2_BK>(ctx, kernel, A, B, C, k, pc, numBatches);
  }

  bool WinogradGemmDot2::isConfigSupported(
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t sgm,
    int32_t sgn,
    int32_t sgmIter,
    int32_t tm,
    int32_t tn,
    int32_t subgroupSize) {
    if(workgroupSize <= 0 || bm <= 0 || bn <= 0 || sgm <= 0 || sgn <= 0)
      return false;
    if(sgmIter <= 0 || tm <= 0 || tn <= 0 || subgroupSize <= 0)
      return false;
    if(bm % 8 != 0 || bn % 8 != 0)
      return false;
    if(tm % 2 != 0)
      return false;  // TM/2 used in accumulator indexing
    if(bm % sgm != 0 || bn % sgn != 0)
      return false;  // subgroup grid tiles the BM×BN block exactly
    if(sgm % sgmIter != 0)
      return false;
    const int32_t sgSubM = sgm / sgmIter;
    if(sgSubM <= 0 || sgSubM % tm != 0)
      return false;  // tiwr uses (SGSUBM/TM) as a modulo divisor
    const int64_t sgnIterNumer = (int64_t)sgm * sgn;
    const int64_t sgnIterDenom = (int64_t)subgroupSize * tm * tn * sgmIter;
    if(sgnIterNumer % sgnIterDenom != 0)
      return false;
    const int64_t sgnIter64 = sgnIterNumer / sgnIterDenom;
    if(sgnIter64 <= 0 || sgnIter64 > INT32_MAX)
      return false;
    const int32_t sgnIter = (int32_t)sgnIter64;
    if(sgn % sgnIter != 0)
      return false;
    const int32_t sgSubN = sgn / sgnIter;
    if(sgSubN <= 0 || sgSubN % tn != 0)
      return false;
    // Every SGM×SGN subgroup region in the BM×BN tile needs its own subgroup: subgroup_c is
    // computed as subgroup_i/(BM/SGM), so too few subgroups leaves the right N-columns
    // unwritten. Require exactly one subgroup per region.
    if((int64_t)workgroupSize != (int64_t)(bm / sgm) * (bn / sgn) * subgroupSize)
      return false;
    return true;
  }

  WinogradGemmDot2::WinogradGemmDot2(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmDot2::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmDot2::sharedParams() {
    // No tunable params for now — fixed at ggml defaults.
    static const std::array<TunableParam, 0> tab = {{}};
    return tab;
  }

  ArrayView<TunableParam> WinogradGemmDot2::params() const {
    return sharedParams();
  }

  bool WinogradGemmDot2::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.dot2WorkgroupSize,
         cfg.dot2BM,
         cfg.dot2BN,
         cfg.dot2SGM,
         cfg.dot2SGN,
         cfg.dot2SGMIter,
         cfg.dot2TM,
         cfg.dot2TN,
         cfg.dot2SubgroupSize))
      return false;
    // Shared tiles: buf_a[BM*(BK+4)] + buf_b[BN*(BK+4)] scalar float16_t (2 bytes).
    // BK is fixed in the shader (#define BK matches VulkanKernels::DOT2_BK); +4 is the bank-conflict pad.
    if(dot2SharedBytes(cfg.dot2BM, cfg.dot2BN) > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.dot2WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmDot2AccF16 --------

  ComputeKernel WinogradGemmDot2AccF16::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    return buildWinogradGemmDot2Twin<true>(device, cache, cfg, k, requiredSubgroupSize);
  }

  bool WinogradGemmDot2AccF16::isConfigSupported(
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t sgm,
    int32_t sgn,
    int32_t sgmIter,
    int32_t tm,
    int32_t tn,
    int32_t subgroupSize) {
    return WinogradGemmDot2::isConfigSupported(workgroupSize, bm, bn, sgm, sgn, sgmIter, tm, tn, subgroupSize);
  }

  void WinogradGemmDot2AccF16::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    WinogradGemmDot2::dispatch(ctx, kernel, A, B, C, k, pc, numBatches);
  }

  WinogradGemmDot2AccF16::WinogradGemmDot2AccF16(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmDot2AccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmDot2AccF16::sharedParams() {
    return WinogradGemmDot2::sharedParams();
  }

  ArrayView<TunableParam> WinogradGemmDot2AccF16::params() const {
    return sharedParams();
  }

  bool WinogradGemmDot2AccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.dot2AccF16WorkgroupSize,
         cfg.dot2AccF16BM,
         cfg.dot2AccF16BN,
         cfg.dot2AccF16SGM,
         cfg.dot2AccF16SGN,
         cfg.dot2AccF16SGMIter,
         cfg.dot2AccF16TM,
         cfg.dot2AccF16TN,
         cfg.dot2AccF16SubgroupSize))
      return false;
    if(dot2SharedBytes(cfg.dot2AccF16BM, cfg.dot2AccF16BN) > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.dot2AccF16WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedDot2Nhwc --------
  void GemmStridedDot2Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<8>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  void GemmStridedDot2AccF16Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<8>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  // Native-NHWC companion to GemmStridedDot2. Compute loop, register tiling and
  // spec-constant layout are identical to GemmStridedDot2 — only the shader
  // source differs (NHWC A/C addressing), so build() mirrors
  // GemmStridedDot2::build() exactly apart from the shader bytes.

  ComputeKernel GemmStridedDot2Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t packedB,
    int32_t kAligned,
    int32_t k,
    bool addToOutput,
    uint32_t requiredSubgroupSize) {
    return buildGemmStridedDot2NhwcTwin<false>(
      device, cache, cfg, aligned, packedB, kAligned, k, addToOutput, requiredSubgroupSize);
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedDot2AccF16Nhwc --------
  // Native-NHWC companion to GemmStridedDot2AccF16 (OpFDot2MixAcc16VALVE).

  ComputeKernel GemmStridedDot2AccF16Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t packedB,
    int32_t kAligned,
    int32_t k,
    bool addToOutput,
    uint32_t requiredSubgroupSize) {
    return buildGemmStridedDot2NhwcTwin<true>(
      device, cache, cfg, aligned, packedB, kAligned, k, addToOutput, requiredSubgroupSize);
  }

  // bench() defined in vulkantuner.cpp.

  // Winograd coopmat uses packed half8 shared staging over even half4 strides:
  // buf_a[BM*((BK/4+2)/2)] + buf_b[BK*((BN/4+2)/2)] uvec4. Both divisions
  // truncate in the shader, so keep them nested here rather than folding to
  // /8 -- for bk or bn not a multiple of 8 the two forms disagree. The accf32
  // and accf16 variants declare identical shared layouts (only the C fragment
  // type differs, and that lives in registers), so one helper serves both.
  static size_t winogradCoopmatSharedBytes(int32_t bm, int32_t bn, int32_t bk) {
    constexpr size_t packedPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    size_t aStrideV8 = ((size_t)bk / 4 + packedPadWords) / 2;
    size_t bStrideV8 = ((size_t)bn / 4 + packedPadWords) / 2;
    return ((size_t)bm * aStrideV8 + (size_t)bk * bStrideV8) * 16;
  }

  // -------- WinogradGemmCoopmat1 --------

  ComputeKernel WinogradGemmCoopmat1::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    return buildWinogradGemmCoopmat1Twin<false>(device, cache, cfg, k, requiredSubgroupSize);
  }

  void WinogradGemmCoopmat1::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchWinogradGemmTile<8>(ctx, kernel, A, B, C, k, pc, numBatches);
  }

  bool WinogradGemmCoopmat1::isConfigSupported(
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t sgm,
    int32_t sgn,
    int32_t tm,
    int32_t tn,
    int32_t tk,
    int32_t subgroupSize) {
    if(tm % 4 != 0 || tn % 8 != 0 || tk % 8 != 0)
      return false;  // packed half8 coopMatLoad offsets must be fragment-aligned
    if(bk % 8 != 0)
      return false;  // A is transported as packed half8 rows before shared staging
    return coopmat1GemmConfigSupported(workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize);
  }

  WinogradGemmCoopmat1::WinogradGemmCoopmat1(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmat1::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmat1::sharedParams() {
    static const std::array<TunableParam, 0> tab = {{}};
    return tab;
  }

  ArrayView<TunableParam> WinogradGemmCoopmat1::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmat1::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
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
      return false;
    size_t sharedBytes = winogradCoopmatSharedBytes(cfg.coopmat1BM, cfg.coopmat1BN, cfg.coopmat1BK);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat1WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat1Nhwc --------
  void GemmStridedCoopmat1Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<8>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  void GemmStridedCoopmat1AccF16Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<8>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  // -------- WinogradGemmCoopmat1AccF16 --------

  ComputeKernel WinogradGemmCoopmat1AccF16::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t k,
    uint32_t requiredSubgroupSize) {
    return buildWinogradGemmCoopmat1Twin<true>(device, cache, cfg, k, requiredSubgroupSize);
  }

  WinogradGemmCoopmat1AccF16::WinogradGemmCoopmat1AccF16(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmat1AccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmat1AccF16::sharedParams() {
    return WinogradGemmCoopmat1::sharedParams();
  }

  ArrayView<TunableParam> WinogradGemmCoopmat1AccF16::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmat1AccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
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
      return false;
    size_t sharedBytes = winogradCoopmatSharedBytes(cfg.coopmat1AccF16BM, cfg.coopmat1AccF16BN, cfg.coopmat1AccF16BK);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat1AccF16WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat1AccF16 --------
  ComputeKernel GemmStridedCoopmat1Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t k,
    uint32_t requiredSubgroupSize,
    bool addToOutput) {
    return buildGemmStridedCoopmat1NhwcTwin<false>(device, cache, cfg, aligned, k, requiredSubgroupSize, addToOutput);
  }

  ComputeKernel GemmStridedCoopmat1AccF16Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t k,
    uint32_t requiredSubgroupSize,
    bool addToOutput) {
    return buildGemmStridedCoopmat1NhwcTwin<true>(device, cache, cfg, aligned, k, requiredSubgroupSize, addToOutput);
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmCoopmat2 --------

  ComputeKernel
  WinogradGemmCoopmat2::build(VkDevice device, VkPipelineCache cache, const VulkanTuneParams& cfg, int32_t k) {
    return buildWinogradGemmCoopmat2Twin<false>(device, cache, cfg, k);
  }

  void WinogradGemmCoopmat2::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchWinogradGemmTile<8>(ctx, kernel, A, B, C, k, pc, numBatches);
  }

  bool WinogradGemmCoopmat2::isConfigSupported(int32_t workgroupSize, int32_t bm, int32_t bn, int32_t bk) {
    if(workgroupSize <= 0 || bm <= 0 || bn <= 0 || bk <= 0)
      return false;
    if(bm % 8 != 0 || bn % 8 != 0 || bk % 8 != 0)
      return false;
    return true;
  }

  WinogradGemmCoopmat2::WinogradGemmCoopmat2(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmat2::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmat2::sharedParams() {
    static const std::array<TunableParam, 0> tab = {{}};
    return tab;
  }

  ArrayView<TunableParam> WinogradGemmCoopmat2::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmat2::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.coopmat2WorkgroupSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat2WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat2Nhwc --------
  void GemmStridedCoopmat2Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<4>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  void GemmStridedCoopmat2AccF16Nhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    dispatchGemmStridedTile<4>(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  // -------- WinogradGemmCoopmat2AccF16 --------

  ComputeKernel
  WinogradGemmCoopmat2AccF16::build(VkDevice device, VkPipelineCache cache, const VulkanTuneParams& cfg, int32_t k) {
    return buildWinogradGemmCoopmat2Twin<true>(device, cache, cfg, k);
  }

  WinogradGemmCoopmat2AccF16::WinogradGemmCoopmat2AccF16(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmat2AccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmat2AccF16::sharedParams() {
    return WinogradGemmCoopmat2::sharedParams();
  }

  ArrayView<TunableParam> WinogradGemmCoopmat2AccF16::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmat2AccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.coopmat2AccF16WorkgroupSize, cfg.coopmat2AccF16BM, cfg.coopmat2AccF16BN, cfg.coopmat2AccF16BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat2AccF16WorkgroupSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat2AccF16 --------
  ComputeKernel GemmStridedCoopmat2Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t kAligned,
    int32_t k,
    bool addToOutput) {
    return buildGemmStridedCoopmat2NhwcTwin<false>(device, cache, cfg, aligned, kAligned, k, addToOutput);
  }

  ComputeKernel GemmStridedCoopmat2AccF16Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t aligned,
    int32_t kAligned,
    int32_t k,
    bool addToOutput) {
    return buildGemmStridedCoopmat2NhwcTwin<true>(device, cache, cfg, aligned, kAligned, k, addToOutput);
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmDirectFP32 --------

  ComputeKernel GemmDirectFP32::build(VkDevice device, VkPipelineCache cache, const VulkanTuneParams& cfg, int32_t k) {
    const int32_t localSizeX = cfg.gemmDirectLocalSizeX;
    const int32_t localSizeY = cfg.gemmDirectLocalSizeY;
    testAssert(isConfigSupported(localSizeX, localSizeY));
    testAssert(k > 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(3);
    auto specDat = makeSpecData({(uint32_t)localSizeX, (uint32_t)localSizeY, (uint32_t)k});
    ComputeKernel kernel =
      ComputeKernel::build(device, gemm_direct_fp32, gemm_direct_fp32_size, 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.launch.data = LaunchProfile::GemmK{(uint32_t)k};
    kernel.debugName = "GemmDirectFP32";
    kernel.debugFp16 = false;
    return kernel;
  }

  void GemmDirectFP32::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* weight,
    VulkanBuffer* output,
    int k,
    const PC& pc) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    testAssert(k > 0);
    const auto& launch = requireLaunchProfile<LaunchProfile::GemmK>(kernel);
    testAssert(launch.kSpec == 0 || k == (int)launch.kSpec);
    kernel.dispatch(
      ctx,
      {input, weight, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.M, kernel.localSizeX),
      divUpU32(pc.N, kernel.localSizeY),
      1u);
  }

  ArrayView<TunableParam> GemmDirectFP32::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_DIRECT
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> GemmDirectFP32::params() const {
    return sharedParams();
  }

  GemmDirectFP32::GemmDirectFP32(int M, int N, int K) : problemM(M), problemN(N), problemK(K) {}

  double GemmDirectFP32::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(1, problemM, problemN, problemK);
  }

  bool GemmDirectFP32::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  bool GemmDirectFP32::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.gemmDirectLocalSizeX, cfg.gemmDirectLocalSizeY))
      return false;
    if(!workGroupFits((uint32_t)cfg.gemmDirectLocalSizeX, (uint32_t)cfg.gemmDirectLocalSizeY, 1u, lim))
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- AttentionTiledNhwc --------

  ComputeKernel AttentionTiledNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    const VulkanTuneParams& cfg,
    int seqLen,
    int headDim,
    int vHeadDim,
    bool useRope,
    bool learnableRope) {
    const int blockQ = cfg.attnNhwcBlockQ;
    const int blockKV = cfg.attnNhwcBlockKV;
    const int qPerThread = cfg.attnNhwcQPerThread;
    testAssert(AttentionTiled::isConfigSupported(blockQ, blockKV, qPerThread));
    testAssert(seqLen > 0);
    if(useRope)
      testAssert((headDim % 2) == 0);
    if(!useRope)
      testAssert(!learnableRope);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(9);
    auto specDat = makeSpecData(
      {(uint32_t)blockQ,
       (uint32_t)blockKV,
       (uint32_t)qPerThread,
       (uint32_t)headDim,
       (uint32_t)vHeadDim,
       fp16 ? 1u : 0u,
       useRope ? 1u : 0u,
       learnableRope ? 1u : 0u,
       (uint32_t)seqLen});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_attention_tiled_nhwc,
      transformer_attention_tiled_nhwc_size,
      useRope ? 12u : (fp16 ? 10u : 5u),
      sizeof(PC),
      specMap,
      specDat,
      cache);
    if(useRope) {
      kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6}
                               : std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 5, 6};
    } else {
      kernel.bindingMap =
        fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    }
    kernel.localSizeX = (uint16_t)blockQ;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::AttentionTiledQ{(uint16_t)qPerThread, (uint32_t)seqLen};
    kernel.debugName = "AttentionTiledNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void AttentionTiledNhwc::dispatch(
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
    const PC& pc) {
    const auto& launch = requireLaunchProfile<LaunchProfile::AttentionTiledQ>(kernel);
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY == 1 && launch.qPerThread > 0);
    testAssert(seqLen > 0 && (launch.seqLenSpec == 0 || seqLen == (int)launch.seqLenSpec));
    testAssert(pc.numKVHeads > 0 && pc.numHeads % pc.numKVHeads == 0);
    uint32_t gx = divUpU32((uint32_t)seqLen, (uint32_t)kernel.localSizeX * (uint32_t)launch.qPerThread);
    if(kernel.bindingMap.size() == 5 || kernel.bindingMap.size() == 10)
      kernel.dispatch(ctx, {Q, K, V, output, mask}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    else {
      testAssert(kernel.bindingMap.size() == 12 && cos != nullptr && sin != nullptr);
      kernel.dispatch(ctx, {Q, K, V, output, mask, cos, sin}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    }
  }

  // -------- AttentionCoopmat1Nhwc --------

  bool AttentionCoopmat1Nhwc::isConfigSupported(
    int workgroupSize,
    int blockQ,
    int blockKV,
    int tm,
    int tn,
    int tk,
    int subgroupSize,
    int headDim,
    int vHeadDim) {
    if(
      workgroupSize <= 0 || blockQ <= 0 || blockKV <= 0 || tm <= 0 || tn <= 0 || tk <= 0 || subgroupSize <= 0 ||
      headDim <= 0 || vHeadDim <= 0)
      return false;
    if(blockQ != workgroupSize)
      return false;
    if(
      workgroupSize % subgroupSize != 0 || blockQ % tm != 0 || blockKV % tn != 0 || blockKV % tk != 0 ||
      blockKV % 8 != 0)
      return false;
    // All operand staging is uvec4 (half8) wide, so every head dim must be a
    // whole number of half8 words. TM/TN/TK likewise, since fragment offsets
    // within a row are computed in uvec4 units.
    if(headDim % 8 != 0 || vHeadDim % 8 != 0 || tk % 8 != 0 || tn % 8 != 0)
      return false;
    // Each subgroup owns a contiguous chunk of Q-strips of TM rows, so the total
    // number of Q-strips must be divisible by the number of subgroups.
    int numSubgroups = workgroupSize / subgroupSize;
    int numQStrips = blockQ / tm;
    if(numQStrips % numSubgroups != 0)
      return false;
    // The output accumulator is a coopmat fragment, so the softmax origin used
    // to rescale it can only be subgroup-uniform. That is consistent only if the
    // Q rows a subgroup's fragments cover are exactly the rows its lanes own --
    // i.e. blockQ / numSubgroups == subgroupSize. Combined with blockQ == workgroupSize this is
    // implied, but assert it explicitly since the shader depends on it.
    if(blockQ / numSubgroups != subgroupSize)
      return false;
    // Also require BLOCK_KV % tk == 0 so the P*V k-loop fully unrolls.
    // Bound fragment arrays and shader compile pressure. The tuner currently
    // proposes only 32/64 tiles, but keep the predicate defensive for loaded
    // tune files.
    int vHeadPad = ((vHeadDim + tn - 1) / tn) * tn;
    int stripsPerSubgroup = numQStrips / numSubgroups;
    int vFragsPerStrip = vHeadPad / tn;
    int fragsPerSubgroup = stripsPerSubgroup * vFragsPerStrip;
    return numSubgroups > 0 && fragsPerSubgroup <= 16;
  }

  bool AttentionCoopmat1Nhwc::isModeConfigSupported(
    int directKV,
    int kvChunkCount,
    int splitKDirectKV,
    int splitKKVChunkCount,
    int splitKCutoffBatch) {
    return (directKV == 0 || directKV == 1) && kvChunkCount >= 1 && kvChunkCount <= 8 &&
           (splitKDirectKV == 0 || splitKDirectKV == 1) && splitKKVChunkCount >= 1 && splitKKVChunkCount <= 8 &&
           splitKCutoffBatch >= 0;
  }

  size_t AttentionCoopmat1Nhwc::sharedBytes(
    int blockQ,
    int blockKV,
    int tk,
    int tn,
    int headDim,
    int vHeadDim,
    bool useRope,
    bool directKV,
    bool maintenance1) {
    int headPad = ((headDim + tk - 1) / tk) * tk;
    int vHeadPad = ((vHeadDim + tn - 1) / tn) * tn;
    // Mirrors the shader's shared layout exactly. In the portable shader Q and P
    // share aPacked (Q is dead once qFrag is in registers) and the fp32 scratch
    // is both the score tile and the epilogue accumulator spill. The
    // maintenance1 shaders keep scores and probabilities in cooperative-matrix
    // registers, so aPacked stages Q alone and scratch is only the spill.
    int aColsV8 = maintenance1 ? headPad / 8 : std::max(headPad / 8, blockKV / 8);
    size_t aPackedV8 = (size_t)blockQ * (aColsV8 + 1);
    // K and V reuse one packed shared tile. Direct V still needs every K row
    // when RoPE is fused in the shader; direct no-RoPE uses one minimal unused
    // row.
    int kvColsV8 = directKV ? headPad / 8 : std::max(headPad / 8, vHeadPad / 8);
    size_t kvTileV8 = (size_t)(directKV && !useRope ? 1 : blockKV) * (kvColsV8 + 1);
    size_t scratchFloat = (size_t)blockQ * ((maintenance1 ? vHeadPad : std::max(blockKV, vHeadPad)) + 1);
    size_t maskFloat = blockKV;
    // maintenanceRowReduce, maintenanceRowOrigin, maintenanceRowActive.
    size_t maintenanceRowsFloat = maintenance1 ? 3 * (size_t)blockQ : 0;
    return (aPackedV8 + kvTileV8) * 16 + (scratchFloat + maskFloat + maintenanceRowsFloat) * sizeof(float);
  }

  ComputeKernel AttentionCoopmat1Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int seqLen,
    int headDim,
    int vHeadDim,
    bool useRope,
    bool learnableRope,
    uint32_t requiredSubgroupSize,
    int kvChunkCount,
    bool maintenance1) {
    // Callers resolve the maintenance1 field set into the base attnNhwcCoopmat1*
    // fields (a local remap in vulkanlayers.cpp; the tuner bench drives
    // maintenance1 candidates through the base fields too), so this build reads
    // the base or split-K set purely by kvChunkCount. maintenance1 only selects
    // the shader variant below.
    const bool splitK = kvChunkCount > 1;
    const int32_t workgroupSize = splitK ? cfg.attnNhwcCoopmat1SplitKWorkgroupSize : cfg.attnNhwcCoopmat1WorkgroupSize;
    const int32_t blockQ = splitK ? cfg.attnNhwcCoopmat1SplitKBlockQ : cfg.attnNhwcCoopmat1BlockQ;
    const int32_t blockKV = splitK ? cfg.attnNhwcCoopmat1SplitKBlockKV : cfg.attnNhwcCoopmat1BlockKV;
    const int32_t tm = splitK ? cfg.attnNhwcCoopmat1SplitKTM : cfg.attnNhwcCoopmat1TM;
    const int32_t tn = splitK ? cfg.attnNhwcCoopmat1SplitKTN : cfg.attnNhwcCoopmat1TN;
    const int32_t tk = splitK ? cfg.attnNhwcCoopmat1SplitKTK : cfg.attnNhwcCoopmat1TK;
    const int32_t subgroupSize = splitK ? cfg.attnNhwcCoopmat1SplitKSubgroupSize : cfg.attnNhwcCoopmat1SubgroupSize;
    const int32_t directKVField = splitK ? cfg.attnNhwcCoopmat1SplitKDirectKV : cfg.attnNhwcCoopmat1DirectKV;
    const bool directKV = directKVField != 0 && seqLen % blockKV == 0;
    testAssert(isConfigSupported(workgroupSize, blockQ, blockKV, tm, tn, tk, subgroupSize, headDim, vHeadDim));
    testAssert(seqLen > 0 && (uint32_t)subgroupSize == requiredSubgroupSize);
    testAssert(!learnableRope || useRope);
    testAssert(kvChunkCount >= 1);
    if(useRope)
      testAssert((headDim % 2) == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    // Direct global loads are valid only when every KV tile is fully in bounds.
    // Keep the assertion here so callers cannot accidentally enable the
    // specialization for a tail tile that would otherwise read past the buffer.
    testAssert(!directKV || seqLen % blockKV == 0);
    auto specMap = makeSpecMap(splitK ? 14 : 13);
    auto specDat = makeSpecData({
      (uint32_t)workgroupSize,
      (uint32_t)blockQ,
      (uint32_t)blockKV,
      (uint32_t)tm,
      (uint32_t)tn,
      (uint32_t)tk,
      (uint32_t)subgroupSize,
      (uint32_t)headDim,
      (uint32_t)vHeadDim,
      useRope ? 1u : 0u,
      learnableRope ? 1u : 0u,
      (uint32_t)seqLen,
      directKV ? 1u : 0u,
    });
    if(splitK)
      specDat.push_back((uint32_t)kvChunkCount);
    if(maintenance1) {
#if !defined(KATAGO_VULKAN_HAS_COOPMAT_MAINTENANCE1_SHADERS)
      testAssert(false && "maintenance1 attention shader was not compiled");
      return ComputeKernel();
#endif
    }
#if defined(KATAGO_VULKAN_HAS_COOPMAT_MAINTENANCE1_SHADERS)
    const uint32_t* shader =
      maintenance1 ? (splitK ? transformer_attention_coopmat1_maintenance1_split_k_nhwc
                             : transformer_attention_coopmat1_maintenance1_nhwc)
                   : (splitK ? transformer_attention_coopmat1_split_k_nhwc : transformer_attention_coopmat1_nhwc);
    const size_t shaderSize = maintenance1 ? (splitK ? transformer_attention_coopmat1_maintenance1_split_k_nhwc_size
                                                     : transformer_attention_coopmat1_maintenance1_nhwc_size)
                                           : (splitK ? transformer_attention_coopmat1_split_k_nhwc_size
                                                     : transformer_attention_coopmat1_nhwc_size);
#else
    const uint32_t* shader = splitK ? transformer_attention_coopmat1_split_k_nhwc : transformer_attention_coopmat1_nhwc;
    const size_t shaderSize =
      splitK ? transformer_attention_coopmat1_split_k_nhwc_size : transformer_attention_coopmat1_nhwc_size;
#endif
    ComputeKernel kernel = ComputeKernel::build(
      device,
      shader,
      shaderSize,
      splitK ? 9u : (useRope ? 7u : 5u),
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups=*/true,
      requiredSubgroupSize,
      "AttentionCoopmat1AccF32Nhwc");
    if(splitK)
      kernel.bindingMap =
        useRope ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8} : std::vector<uint8_t>{0, 1, 2, 3, 4, 4, 4, 5, 6};
    else
      kernel.bindingMap = useRope ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::AttentionAccelQ{(uint16_t)blockQ, (uint32_t)seqLen, (uint32_t)kvChunkCount};
    kernel.debugName = splitK ? "AttentionCoopmat1SplitKNhwc" : "AttentionCoopmat1AccF32Nhwc";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)directKV;
    (void)maintenance1;
    return failUncompiledCoopmat();
#endif
  }

  size_t AttentionCoopmat1Nhwc::partialsBytes(int seqLen, int numBH, int kvChunkCount, int vHeadDim) {
    return (size_t)numBH * kvChunkCount * seqLen * vHeadDim * sizeof(float);
  }

  size_t AttentionCoopmat1Nhwc::statsBytes(int seqLen, int numBH, int kvChunkCount) {
    return (size_t)numBH * kvChunkCount * seqLen * 2 * sizeof(float);
  }

  void AttentionCoopmat1Nhwc::dispatch(
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
    const PC& pc,
    VulkanBuffer* partials,
    VulkanBuffer* stats) {
    const auto& launch = requireLaunchProfile<LaunchProfile::AttentionAccelQ>(kernel);
    testAssert(kernel.localSizeX > 0 && launch.blockQ > 0 && seqLen > 0);
    testAssert(launch.seqLenSpec == 0 || seqLen == (int)launch.seqLenSpec);
    testAssert(pc.numKVHeads > 0 && pc.numHeads % pc.numKVHeads == 0);
    uint32_t gx = divUpU32((uint32_t)seqLen, (uint32_t)launch.blockQ);
    uint32_t gz = launch.kvChunkCount;
    if(kernel.bindingMap.size() == 9) {
      testAssert(partials != nullptr && stats != nullptr);
      if(cos != nullptr && sin != nullptr)
        kernel.dispatch(
          ctx, {Q, K, V, output, mask, cos, sin, partials, stats}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, gz);
      else
        kernel.dispatch(ctx, {Q, K, V, output, mask, partials, stats}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, gz);
    } else if(kernel.bindingMap.size() == 7) {
      testAssert(cos != nullptr && sin != nullptr);
      kernel.dispatch(ctx, {Q, K, V, output, mask, cos, sin}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, gz);
    } else {
      testAssert(kernel.bindingMap.size() == 5);
      kernel.dispatch(ctx, {Q, K, V, output, mask}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, gz);
    }
  }

  // -------- AttentionSplitKResolveNhwc --------

  static bool isConfigSupported(int vHeadDim) {
    return vHeadDim > 0 && vHeadDim % 2 == 0;
  }

  ComputeKernel AttentionSplitKResolveNhwc::build(VkDevice device, VkPipelineCache cache, int vHeadDim) {
    testAssert(isConfigSupported(vHeadDim));
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS) || defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)(vHeadDim / 2), (uint32_t)vHeadDim});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_attention_split_k_resolve_nhwc,
      transformer_attention_split_k_resolve_nhwc_size,
      3u,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups=*/false,
      0,
      "AttentionSplitKResolveNhwc");
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)(vHeadDim / 2);
    kernel.localSizeY = 1;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)vHeadDim;
    testAssert(false && "AttentionSplitKResolveNhwc::build called without compiled coopmat shaders");
    return ComputeKernel();
#endif
  }

  void AttentionSplitKResolveNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* partials,
    VulkanBuffer* stats,
    VulkanBuffer* output,
    int seqLen,
    int numBH,
    const PC& pc) {
    testAssert(partials != nullptr && stats != nullptr && output != nullptr);
    kernel.dispatch(ctx, {partials, stats, output}, &pc, sizeof(PC), (uint32_t)seqLen, (uint32_t)numBH, 1u);
  }

  // -------- AttentionCoopmat2AccF32Nhwc --------

  bool AttentionCoopmat2AccF32Nhwc::isConfigSupported(
    int workgroupSize,
    int blockQ,
    int blockKV,
    int headDim,
    int vHeadDim) {
    return workgroupSize > 0 && blockQ > 0 && blockKV > 0 && blockKV % 8 == 0 && headDim > 0 && headDim % 8 == 0 &&
           vHeadDim > 0 && vHeadDim % 8 == 0;
  }

  size_t AttentionCoopmat2AccF32Nhwc::sharedBytes(int blockQ, int blockKV, int headDim, int vHeadDim) {
    (void)headDim;
    (void)vHeadDim;
    return ((size_t)blockQ + (size_t)blockKV) * sizeof(uint32_t);
  }

  ComputeKernel AttentionCoopmat2AccF32Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int seqLen,
    int headDim,
    int vHeadDim,
    bool useRope,
    bool learnableRope) {
    const int workgroupSize = cfg.attnNhwcCoopmat2WorkgroupSize;
    const int blockQ = cfg.attnNhwcCoopmat2BlockQ;
    const int blockKV = cfg.attnNhwcCoopmat2BlockKV;
    testAssert(isConfigSupported(workgroupSize, blockQ, blockKV, headDim, vHeadDim));
    testAssert(seqLen > 0 && (!learnableRope || useRope));
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    using namespace VulkanShaders;
    // Undefined clamp mode is legal only when every Q/KV tile is wholly
    // inside the sequence. The tensor strides remain canonical in both paths.
    const bool alignedTiles = seqLen % blockQ == 0 && seqLen % blockKV == 0;
    auto specMap = makeSpecMap(9);
    auto specDat = makeSpecData(
      {(uint32_t)workgroupSize,
       (uint32_t)blockQ,
       (uint32_t)blockKV,
       (uint32_t)headDim,
       (uint32_t)vHeadDim,
       useRope ? 1u : 0u,
       learnableRope ? 1u : 0u,
       (uint32_t)seqLen,
       alignedTiles ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_attention_coopmat2_accf32_nhwc,
      transformer_attention_coopmat2_accf32_nhwc_size,
      useRope ? 7u : 5u,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      false,
      0u,
      "AttentionCoopmat2AccF32Nhwc");
    kernel.bindingMap = useRope ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.launch.data = LaunchProfile::AttentionAccelQ{(uint16_t)blockQ, (uint32_t)seqLen};
    kernel.debugName = "AttentionCoopmat2AccF32Nhwc";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)workgroupSize;
    (void)blockQ;
    (void)blockKV;
    (void)seqLen;
    (void)headDim;
    (void)vHeadDim;
    (void)useRope;
    (void)learnableRope;
    testAssert(false && "AttentionCoopmat2AccF32Nhwc::build called without compiled coopmat2 shaders");
    return ComputeKernel();
#endif
  }

  void AttentionCoopmat2AccF32Nhwc::dispatch(
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
    const PC& pc) {
    const auto& launch = requireLaunchProfile<LaunchProfile::AttentionAccelQ>(kernel);
    testAssert(kernel.localSizeX > 0 && launch.blockQ > 0 && seqLen > 0);
    const uint32_t gx = divUpU32((uint32_t)seqLen, (uint32_t)launch.blockQ);
    if(kernel.bindingMap.size() == 5)
      kernel.dispatch(ctx, {Q, K, V, output, mask}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    else {
      testAssert(kernel.bindingMap.size() == 7 && cos != nullptr && sin != nullptr);
      kernel.dispatch(ctx, {Q, K, V, output, mask, cos, sin}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    }
  }

  // -------- AttentionDot2AccF32Nhwc --------

  bool
  AttentionDot2AccF32Nhwc::isConfigSupported(int workgroupSize, int blockQ, int blockKV, int headDim, int vHeadDim) {
    return workgroupSize > 0 && blockQ == workgroupSize && blockKV >= 4 && blockKV <= 64 && blockKV % 4 == 0 &&
           headDim > 0 && headDim % 8 == 0 && vHeadDim > 0 && vHeadDim % 8 == 0;
  }

  size_t AttentionDot2AccF32Nhwc::sharedBytes(int blockKV, int headDim, int vHeadDim) {
    // kTile is packed half8 with one padding word per row. vTile is transposed
    // scalar FP16 with four padding elements per dimension. kvMask is FP32.
    size_t kPackedV8 = (size_t)blockKV * ((size_t)headDim / 8 + 1);
    size_t vHalf = (size_t)vHeadDim * ((size_t)blockKV + 4);
    return kPackedV8 * 16 + vHalf * sizeof(uint16_t) + (size_t)blockKV * sizeof(float);
  }

  ComputeKernel AttentionDot2AccF32Nhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int seqLen,
    int headDim,
    int vHeadDim,
    bool useRope,
    bool learnableRope) {
    const int workgroupSize = cfg.attnNhwcDot2WorkgroupSize;
    const int blockQ = cfg.attnNhwcDot2BlockQ;
    const int blockKV = cfg.attnNhwcDot2BlockKV;
    testAssert(isConfigSupported(workgroupSize, blockQ, blockKV, headDim, vHeadDim));
    testAssert(seqLen > 0 && (!learnableRope || useRope));
    if(useRope)
      testAssert((headDim % 2) == 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(8);
    auto specDat = makeSpecData({
      (uint32_t)workgroupSize,
      (uint32_t)blockQ,
      (uint32_t)blockKV,
      (uint32_t)headDim,
      (uint32_t)vHeadDim,
      useRope ? 1u : 0u,
      learnableRope ? 1u : 0u,
      (uint32_t)seqLen,
    });
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_attention_dot2_accf32_nhwc,
      transformer_attention_dot2_accf32_nhwc_size,
      useRope ? 7u : 5u,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      false,
      0u,
      "AttentionDot2AccF32Nhwc");
    kernel.bindingMap = useRope ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.localSizeX = (uint16_t)workgroupSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::AttentionAccelQ{(uint16_t)blockQ, (uint32_t)seqLen};
    kernel.debugName = "AttentionDot2AccF32Nhwc";
    kernel.debugFp16 = true;
    return kernel;
  }

  void AttentionDot2AccF32Nhwc::dispatch(
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
    const PC& pc) {
    const auto& launch = requireLaunchProfile<LaunchProfile::AttentionAccelQ>(kernel);
    testAssert(kernel.localSizeX > 0 && launch.blockQ > 0 && seqLen > 0);
    testAssert(launch.seqLenSpec == 0 || seqLen == (int)launch.seqLenSpec);
    testAssert(pc.numKVHeads > 0 && pc.numHeads % pc.numKVHeads == 0);
    uint32_t gx = divUpU32((uint32_t)seqLen, (uint32_t)launch.blockQ);
    if(kernel.bindingMap.size() == 5)
      kernel.dispatch(ctx, {Q, K, V, output, mask}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    else {
      testAssert(kernel.bindingMap.size() == 7 && cos != nullptr && sin != nullptr);
      kernel.dispatch(ctx, {Q, K, V, output, mask, cos, sin}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    }
  }

  AttentionDot2AccF32Bench::AttentionDot2AccF32Bench(
    int batchSize,
    int numTokens,
    int headDim,
    int vHeadDim,
    int numHeads,
    int numKVHeads,
    bool useRope,
    bool learnableRope,
    int validTokens)
    : problemBatchSize(batchSize),
      problemNumTokens(numTokens),
      problemValidTokens(validTokens < 0 ? numTokens : validTokens),
      problemHeadDim(headDim),
      problemVHeadDim(vHeadDim),
      problemNumHeads(numHeads),
      problemNumKVHeads(numKVHeads),
      problemUseRope(useRope),
      problemLearnableRope(learnableRope) {}

  size_t AttentionCoopmat1Bench::requiredSharedBytes(const VulkanTuneParams& cfg) const {
    return AttentionCoopmat1Nhwc::sharedBytes(
      cfg.attnNhwcCoopmat1BlockQ,
      cfg.attnNhwcCoopmat1BlockKV,
      cfg.attnNhwcCoopmat1TK,
      cfg.attnNhwcCoopmat1TN,
      problemHeadDim,
      problemVHeadDim,
      problemUseRope,
      cfg.attnNhwcCoopmat1DirectKV != 0 && problemNumTokens % cfg.attnNhwcCoopmat1BlockKV == 0,
      usesMaintenance1());
  }

  bool AttentionDot2AccF32Bench::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const {
    if(!AttentionDot2AccF32Nhwc::isConfigSupported(
         cfg.attnNhwcDot2WorkgroupSize,
         cfg.attnNhwcDot2BlockQ,
         cfg.attnNhwcDot2BlockKV,
         problemHeadDim,
         problemVHeadDim))
      return false;
    if(!workGroupFits((uint32_t)cfg.attnNhwcDot2WorkgroupSize, 1u, 1u, limits))
      return false;
    return AttentionDot2AccF32Nhwc::sharedBytes(cfg.attnNhwcDot2BlockKV, problemHeadDim, problemVHeadDim) <=
           limits.maxComputeSharedMemorySize;
  }

  double AttentionDot2AccF32Bench::estimatedFlopsPerDispatch() const {
    return attentionMatmulFlopsPerDispatch(
      problemBatchSize, problemNumTokens, problemHeadDim, problemVHeadDim, problemNumHeads);
  }

  AttentionCoopmat1Bench::AttentionCoopmat1Bench(
    int batchSize,
    int numTokens,
    int headDim,
    int vHeadDim,
    int numHeads,
    int numKVHeads,
    bool useRope,
    bool learnableRope,
    int validTokens)
    : problemBatchSize(batchSize),
      problemNumTokens(numTokens),
      problemValidTokens(validTokens < 0 ? numTokens : validTokens),
      problemHeadDim(headDim),
      problemVHeadDim(vHeadDim),
      problemNumHeads(numHeads),
      problemNumKVHeads(numKVHeads),
      problemUseRope(useRope),
      problemLearnableRope(learnableRope) {}

  bool AttentionCoopmat1Bench::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const {
    if(!AttentionCoopmat1Nhwc::isConfigSupported(
         cfg.attnNhwcCoopmat1WorkgroupSize,
         cfg.attnNhwcCoopmat1BlockQ,
         cfg.attnNhwcCoopmat1BlockKV,
         cfg.attnNhwcCoopmat1TM,
         cfg.attnNhwcCoopmat1TN,
         cfg.attnNhwcCoopmat1TK,
         cfg.attnNhwcCoopmat1SubgroupSize,
         problemHeadDim,
         problemVHeadDim))
      return false;
    if(!workGroupFits((uint32_t)cfg.attnNhwcCoopmat1WorkgroupSize, 1u, 1u, limits))
      return false;
    return requiredSharedBytes(cfg) <= limits.maxComputeSharedMemorySize;
  }

  double AttentionCoopmat1Bench::estimatedFlopsPerDispatch() const {
    // Two QK^T passes plus one PV pass.
    if(
      problemBatchSize <= 0 || problemNumTokens <= 0 || problemHeadDim <= 0 || problemVHeadDim <= 0 ||
      problemNumHeads <= 0)
      return 0.0;
    double bh = (double)problemBatchSize * problemNumHeads;
    double pairs = (double)problemNumTokens * problemNumTokens;
    return 2.0 * bh * pairs * (2.0 * problemHeadDim + problemVHeadDim);
  }

  AttentionCoopmat2AccF32Bench::AttentionCoopmat2AccF32Bench(
    int batchSize,
    int numTokens,
    int headDim,
    int vHeadDim,
    int numHeads,
    int numKVHeads,
    bool useRope,
    bool learnableRope,
    int validTokens)
    : problemBatchSize(batchSize),
      problemNumTokens(numTokens),
      problemValidTokens(validTokens < 0 ? numTokens : validTokens),
      problemHeadDim(headDim),
      problemVHeadDim(vHeadDim),
      problemNumHeads(numHeads),
      problemNumKVHeads(numKVHeads),
      problemUseRope(useRope),
      problemLearnableRope(learnableRope) {}

  bool AttentionCoopmat2AccF32Bench::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& limits) const {
    return AttentionCoopmat2AccF32Nhwc::isConfigSupported(
             cfg.attnNhwcCoopmat2WorkgroupSize,
             cfg.attnNhwcCoopmat2BlockQ,
             cfg.attnNhwcCoopmat2BlockKV,
             problemHeadDim,
             problemVHeadDim) &&
           workGroupFits((uint32_t)cfg.attnNhwcCoopmat2WorkgroupSize, 1u, 1u, limits) &&
           AttentionCoopmat2AccF32Nhwc::sharedBytes(
             cfg.attnNhwcCoopmat2BlockQ, cfg.attnNhwcCoopmat2BlockKV, problemHeadDim, problemVHeadDim) <=
             limits.maxComputeSharedMemorySize;
  }

  double AttentionCoopmat2AccF32Bench::estimatedFlopsPerDispatch() const {
    return attentionMatmulFlopsPerDispatch(
      problemBatchSize, problemNumTokens, problemHeadDim, problemVHeadDim, problemNumHeads);
  }

  // ---- AttentionTiled (TunableKernel wrapper around Attention::buildTiled) ----

  AttentionTiled::AttentionTiled(
    int batchSize,
    int numTokens,
    int headDim,
    int vHeadDim,
    int numHeads,
    int numKVHeads,
    bool useRope,
    bool learnableRope,
    int validTokens)
    : problemBatchSize(batchSize),
      problemNumTokens(numTokens),
      problemValidTokens(validTokens < 0 ? numTokens : validTokens),
      problemHeadDim(headDim),
      problemVHeadDim(vHeadDim),
      problemNumHeads(numHeads),
      problemNumKVHeads(numKVHeads),
      problemUseRope(useRope),
      problemLearnableRope(learnableRope) {
    testAssert(!problemLearnableRope || problemUseRope);
  }

  double AttentionTiled::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return attentionMatmulFlopsPerDispatch(
      problemBatchSize, problemNumTokens, problemHeadDim, problemVHeadDim, problemNumHeads);
  }

  ArrayView<TunableParam> AttentionTiled::params() const {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_TILED
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }

  bool AttentionTiled::isConfigSupported(int32_t blockQ, int32_t blockKV, int32_t qPerThread) {
    // transformer_attention_tiled_nhwc.glsl loads kMaskTile with one write per localIdx,
    // so every KV-lane must have a writer in the workgroup.
    return blockQ > 0 && blockKV > 0 && qPerThread > 0 && blockKV <= blockQ;
  }

  bool AttentionTiled::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    const int blockQ = cfg.attnNhwcBlockQ;
    const int blockKV = cfg.attnNhwcBlockKV;
    const int qPerThread = cfg.attnNhwcQPerThread;
    if(!isConfigSupported(blockQ, blockKV, qPerThread))
      return false;
    if(problemUseRope && (problemHeadDim % 2) != 0)
      return false;
    if(!workGroupFits((uint32_t)blockQ, 1u, 1u, lim))
      return false;
    // kTile, vTile, kMaskTile, and blockMaskMax[1].
    size_t sharedBytes = ((size_t)blockKV * ((size_t)problemHeadDim + problemVHeadDim + 1) + 1) * sizeof(float);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp.

  // -------- SwiGLU --------

  bool SwiGLU::isConfigSupported(int32_t localSizeX) {
    return localSizeX == 64 || localSizeX == 128 || localSizeX == 256 || localSizeX == 512;
  }

  ComputeKernel SwiGLU::build(VkDevice device, VkPipelineCache cache, bool fp16, const VulkanTuneParams& cfg) {
    const int32_t localSizeX = cfg.swiGLULocalSizeX;
    testAssert(isConfigSupported(localSizeX));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap({0, 1});
    auto specDat = makeSpecData({fp16 ? 1u : 0u, (uint32_t)localSizeX});
    ComputeKernel kernel = ComputeKernel::build(
      device, transformer_swiglu, transformer_swiglu_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1, 2} : std::vector<uint8_t>{0, 1, 2};
    kernel.debugName = "SwiGLU";
    kernel.debugFp16 = fp16;
    kernel.localSizeX = (uint16_t)localSizeX;
    return kernel;
  }

  void SwiGLU::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* main,
    VulkanBuffer* gate,
    VulkanBuffer* output,
    const PC& pc) {
    testAssert(pc.numElements % SwiGLU::ELEMENTS_PER_VEC == 0);
    uint32_t numVec4 = (uint32_t)(pc.numElements / SwiGLU::ELEMENTS_PER_VEC);
    kernel.dispatch(ctx, {main, gate, output}, &pc, sizeof(PC), divUpU32(numVec4, kernel.localSizeX), 1u, 1u);
  }

  ArrayView<TunableParam> SwiGLU::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_SWIGLU
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> SwiGLU::params() const {
    return sharedParams();
  }
  SwiGLU::SwiGLU(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}
  bool SwiGLU::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    return isConfigSupported(cfg.swiGLULocalSizeX) && workGroupFits((uint32_t)cfg.swiGLULocalSizeX, 1u, 1u, lim);
  }
  double SwiGLU::estimatedFlopsPerDispatch(const VulkanTuneParams&) const {
    return 5.0 * (double)problemBatchSize * problemChannels * problemXySize;
  }

  // -------- GPoolReductionNhwc --------

  ComputeKernel
  GPoolReductionNhwc::build(VkDevice device, VkPipelineCache cache, bool fp16, const VulkanTuneParams& cfg) {
    const int xyStride = cfg.gpoolNhwcXystride;
    testAssert(xyStride > 0 && isPow2(xyStride));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)xyStride, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, gpool_mask_nhwc, gpool_mask_nhwc_size, fp16 ? 6 : 4, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 0, 2} : std::vector<uint8_t>{0, 1, 2, 3};
    kernel.debugName = "GPoolReductionNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void GPoolReductionNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx, {input, output, mask, maskSum}, &pc, sizeof(PC), 1u, (uint32_t)pc.numChannels, (uint32_t)maxBatchSize);
  }

  ArrayView<TunableParam> GPoolReductionNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_GPOOL_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> GPoolReductionNhwc::params() const {
    return sharedParams();
  }

  GPoolReductionNhwc::GPoolReductionNhwc(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double GPoolReductionNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    int xyStride = cfg.gpoolNhwcXystride;
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || xyStride <= 0)
      return 0.0;
    double perChannel =
      4.0 * (double)problemXySize + 2.0 * (double)(xyStride - 1) + 5.0;  // sum/max loop, reduction, final scale
    return (double)problemBatchSize * (double)problemChannels * perChannel;
  }

  bool GPoolReductionNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    int xyStride = cfg.gpoolNhwcXystride;
    if(!(xyStride > 0 && isPow2(xyStride)))
      return false;
    if(!workGroupFits((uint32_t)xyStride, 1u, 1u, lim))
      return false;
    if((size_t)xyStride * 2 * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- ValueHeadPoolNhwc --------

  ComputeKernel
  ValueHeadPoolNhwc::build(VkDevice device, VkPipelineCache cache, bool fp16, const VulkanTuneParams& cfg) {
    const int xyStride = cfg.valueHeadPoolNhwcXystride;
    testAssert(xyStride > 0 && isPow2(xyStride));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)xyStride, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, value_head_pool_nhwc, value_head_pool_nhwc_size, fp16 ? 4 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0} : std::vector<uint8_t>{0, 1, 2};
    kernel.debugName = "ValueHeadPoolNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void ValueHeadPoolNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* maskSum,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx, {input, output, maskSum}, &pc, sizeof(PC), 1u, (uint32_t)pc.numChannels, (uint32_t)maxBatchSize);
  }

  ArrayView<TunableParam> ValueHeadPoolNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_VALUE_HEAD_POOL_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> ValueHeadPoolNhwc::params() const {
    return sharedParams();
  }

  ValueHeadPoolNhwc::ValueHeadPoolNhwc(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double ValueHeadPoolNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    int xyStride = cfg.valueHeadPoolNhwcXystride;
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || xyStride <= 0)
      return 0.0;
    double perChannel = (double)problemXySize + (double)(xyStride - 1) + 9.0;  // sum loop, reduction, final features
    return (double)problemBatchSize * (double)problemChannels * perChannel;
  }

  bool ValueHeadPoolNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    int xyStride = cfg.valueHeadPoolNhwcXystride;
    if(!(xyStride > 0 && isPow2(xyStride)))
      return false;
    if(!workGroupFits((uint32_t)xyStride, 1u, 1u, lim))
      return false;
    if((size_t)xyStride * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  ComputeKernel TransformerRMSNormNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    bool useSubgroupVariant,
    uint32_t requiredSubgroupSize) {
    testAssert(useSubgroupVariant == (requiredSubgroupSize > 0));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({2u, 32u, 1u, fp16 ? 1u : 0u});
    const uint32_t* spirvData = useSubgroupVariant ? transformer_rmsnorm_nhwc_subgroup : transformer_rmsnorm_nhwc;
    const size_t spirvWords =
      useSubgroupVariant ? transformer_rmsnorm_nhwc_subgroup_size : transformer_rmsnorm_nhwc_size;
    const bool requireFullSubgroups = useSubgroupVariant;
    ComputeKernel kernel = ComputeKernel::build(
      device,
      spirvData,
      spirvWords,
      fp16 ? 8 : 5,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      requireFullSubgroups,
      requireFullSubgroups ? requiredSubgroupSize : 0u);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    // localSizeX records positions handled per workgroup for dispatch grid math.
    kernel.localSizeX = 2;
    kernel.debugName = useSubgroupVariant ? "TransformerRMSNormNhwc.subgroup" : "TransformerRMSNormNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void TransformerRMSNormNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* gamma,
    VulkanBuffer* beta,
    VulkanBuffer* mask,
    const PC& pc,
    int maxBatchSize) {
    testAssert(kernel.localSizeX > 0);
    kernel.dispatch(
      ctx,
      {input, output, gamma, beta, mask},
      &pc,
      sizeof(PC),
      divUpU32(pc.paddedSpatialSize, kernel.localSizeX),
      (uint32_t)maxBatchSize,
      1u);
  }

  // -------- Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8 --------

  ComputeKernel Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t packedK,
    bool addToOutput) {
    return buildConv3x3ImplicitGemmCoopmat2Vec8Twin<false>(device, cache, cfg, convSize, packedK, addToOutput);
  }

  void Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int convSize,
    const PC& pc,
    int batchSize) {
    testAssert(input && filter && output && input->buffer != output->buffer);
    const auto& tile = requireLaunchProfile<LaunchProfile::GemmDot2Tile>(kernel);
    testAssert(batchSize > 0 && pc.numInChannels > 0 && pc.numInChannels % 8 == 0);
    testAssert(pc.paddedSpatialSize % tile.bm == 0 && pc.numOutChannels % tile.bn == 0);
    testAssert(
      (convSize == 3 || convSize == 5) && pc.numOutChannels == pc.numOutChannelsPadded &&
      convSize * convSize * pc.numInChannels == (int)tile.kSpec);
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      (uint32_t)pc.paddedSpatialSize / tile.bm,
      (uint32_t)pc.numOutChannels / tile.bn,
      (uint32_t)batchSize);
  }

  ComputeKernel Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t packedK,
    bool addToOutput) {
    return buildConv3x3ImplicitGemmCoopmat2Vec8Twin<true>(device, cache, cfg, convSize, packedK, addToOutput);
  }

  void Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int convSize,
    const PC& pc,
    int batchSize) {
    Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::dispatch(ctx, kernel, input, filter, output, convSize, pc, batchSize);
  }

  bool Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t sgm,
    int32_t sgn,
    int32_t tm,
    int32_t tn,
    int32_t tk,
    int32_t subgroupSize) {
    if(!WinogradGemmCoopmat1::isConfigSupported(workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize))
      return false;
    return workgroupSize % (bk / 8) == 0;
  }

  size_t Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::sharedBytes(int32_t bm, int32_t bn, int32_t bk) {
    if(bm <= 0 || bn <= 0 || bk <= 0 || bk % 8 != 0)
      return std::numeric_limits<size_t>::max();
    const size_t aBytes = (size_t)bm * ((size_t)bk / 8 + 1) * 4 * sizeof(uint32_t);
    const size_t bBytes = (size_t)bn * ((size_t)bk + 8) * sizeof(uint16_t);
    return aBytes + bBytes;
  }

  ComputeKernel Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t k,
    bool addToOutput,
    bool slabInsideTap,
    uint32_t requiredSubgroupSize) {
    return buildConv3x3ImplicitGemmCoopmat1Vec8Twin<false>(
      device, cache, cfg, convSize, k, addToOutput, slabInsideTap, requiredSubgroupSize);
  }

  void Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int convSize,
    const PC& pc,
    int batchSize) {
    testAssert(input != nullptr && filter != nullptr && output != nullptr && input->buffer != output->buffer);
    testAssert(batchSize > 0 && pc.batchOffset == 0 && pc.numInChannels > 0 && pc.numInChannels % 8 == 0);
    const auto& tile = requireLaunchProfile<LaunchProfile::GemmDot2Tile>(kernel);
    testAssert(pc.paddedSpatialSize % tile.bm == 0 && pc.numOutChannels % tile.bn == 0);
    testAssert(
      (convSize == 3 || convSize == 5) && pc.numOutChannels == pc.numOutChannelsPadded &&
      convSize * convSize * pc.numInChannels == (int)tile.kSpec);
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      (uint32_t)pc.paddedSpatialSize / tile.bm,
      (uint32_t)pc.numOutChannelsPadded / tile.bn,
      (uint32_t)batchSize);
  }

  bool Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
    int32_t workgroupSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t sgm,
    int32_t sgn,
    int32_t tm,
    int32_t tn,
    int32_t tk,
    int32_t subgroupSize) {
    if(!WinogradGemmCoopmat1AccF16::isConfigSupported(workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize))
      return false;
    return workgroupSize % (bk / 8) == 0;
  }

  size_t Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::sharedBytes(int32_t bm, int32_t bn, int32_t bk) {
    return Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::sharedBytes(bm, bn, bk);
  }

  ComputeKernel Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::build(
    VkDevice device,
    VkPipelineCache cache,
    const VulkanTuneParams& cfg,
    int32_t convSize,
    int32_t k,
    bool addToOutput,
    bool slabInsideTap,
    uint32_t requiredSubgroupSize) {
    return buildConv3x3ImplicitGemmCoopmat1Vec8Twin<true>(
      device, cache, cfg, convSize, k, addToOutput, slabInsideTap, requiredSubgroupSize);
  }

  void Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int convSize,
    const PC& pc,
    int batchSize) {
    Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::dispatch(ctx, kernel, input, filter, output, convSize, pc, batchSize);
  }

  // -------- Conv2dDirectNhwc --------

  ComputeKernel Conv2dDirectNhwc::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({8u, 4u, 4u, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, conv2d_direct_nhwc, conv2d_direct_nhwc_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 2, 1} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = 8;
    kernel.localSizeY = 4;
    kernel.debugName = "Conv2dDirectNhwc";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void Conv2dDirectNhwc::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    const PC& pc) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.nnXLen, kernel.localSizeX),
      divUpU32(pc.nnYLen, kernel.localSizeY),
      (uint32_t)pc.numOutChannels);
  }

  // -------- SpatialRMSNorm --------

  static ComputeKernel buildPass2(
    VkDevice device,
    VkPipelineCache cache,
    int tileSize,
    bool useSubgroupVariant,
    uint32_t requiredSubgroupSize) {
    testAssert(SpatialRMSNormNhwc::isConfigSupported(tileSize));
    testAssert(useSubgroupVariant == (requiredSubgroupSize > 0));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({(uint32_t)tileSize});
    const uint32_t* spirvData =
      useSubgroupVariant ? transformer_spatial_rmsnorm_pass2_subgroup : transformer_spatial_rmsnorm_pass2;
    const size_t spirvWords =
      useSubgroupVariant ? transformer_spatial_rmsnorm_pass2_subgroup_size : transformer_spatial_rmsnorm_pass2_size;
    // Stage-1 subgroupSum in the subgroup variant assumes full subgroups
    // (inactive lanes contribute undefined values otherwise). Only claim
    // requireFullSubgroups when the caller both selects the subgroup shader and
    // has a subgroup size to pin -- i.e. after canUseRMSNormSubgroupVariant.
    const bool requireFullSubgroups = useSubgroupVariant;
    ComputeKernel kernel = ComputeKernel::build(
      device,
      spirvData,
      spirvWords,
      2,
      sizeof(SpatialRMSNormNhwc::Pass2PC),
      specMap,
      specDat,
      cache,
      requireFullSubgroups,
      requireFullSubgroups ? requiredSubgroupSize : 0u);
    // logical: {partials, scalar} — FP32-only shader.
    kernel.bindingMap = {0, 1};
    kernel.localSizeX = (uint16_t)tileSize;
    kernel.debugName = useSubgroupVariant ? "SpatialRMSNorm.pass2.subgroup" : "SpatialRMSNorm.pass2";
    kernel.debugFp16 = false;
    return kernel;
  }

  static ComputeKernel buildPass1(VkDevice device, VkPipelineCache cache, bool fp16, int tileSize) {
    testAssert(SpatialRMSNormNhwc::isConfigSupported(tileSize));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)tileSize, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_spatial_rmsnorm_nhwc_pass1,
      transformer_spatial_rmsnorm_nhwc_pass1_size,
      fp16 ? 5 : 3,
      sizeof(SpatialRMSNormNhwc::Pass1PC),
      specMap,
      specDat,
      cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)tileSize;
    kernel.debugName = "SpatialRMSNormNhwc.pass1";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  static ComputeKernel buildPass3(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)SpatialRMSNormNhwc::APPLY_ELTS_PER_THREAD, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_spatial_rmsnorm_nhwc_pass3,
      transformer_spatial_rmsnorm_nhwc_pass3_size,
      fp16 ? 10 : 7,
      sizeof(SpatialRMSNormNhwc::Pass3PC),
      specMap,
      specDat,
      cache);
    kernel.bindingMap =
      fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 0, 1, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6};
    kernel.debugName = "SpatialRMSNormNhwc.pass3";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  std::array<ComputeKernel, 3> SpatialRMSNormNhwc::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    const VulkanTuneParams& cfg,
    bool useSubgroupVariant,
    uint32_t requiredSubgroupSize) {
    const int tileSize = cfg.spatialRMSNormNhwcTile;
    return {
      buildPass1(device, cache, fp16, tileSize),
      buildPass2(device, cache, tileSize, useSubgroupVariant, requiredSubgroupSize),
      buildPass3(device, cache, fp16)};
  }

  void SpatialRMSNormNhwc::dispatch(
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
    int maxBatchSize) {
    testAssert(kernels[0].localSizeX > 0 && kernels[0].localSizeX == kernels[1].localSizeX);
    const int tileSize = (int)kernels[0].localSizeX;
    const int numSpatialWorkgroups = (pc.numChannels * pc.paddedSpatialSize + tileSize - 1) / tileSize;
    Pass1PC pc1 = {pc.numChannels, pc.paddedSpatialSize, 1, numSpatialWorkgroups};
    kernels[0].dispatch(
      ctx,
      {input, mask, partials},
      &pc1,
      sizeof(Pass1PC),
      (uint32_t)pc1.numSpatialWorkgroups,
      (uint32_t)maxBatchSize,
      1u);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, partials->buffer);
    Pass2PC pc2 = {numSpatialWorkgroups, (numSpatialWorkgroups + tileSize - 1) / tileSize};
    kernels[1].dispatch(ctx, {partials, scalar}, &pc2, sizeof(Pass2PC), 1u, (uint32_t)maxBatchSize, 1u);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, scalar->buffer);
    Pass3PC pc3 = {pc.numChannels, pc.paddedSpatialSize, pc.epsilon};
    const int spatialSize = pc3.numChannels * pc3.paddedSpatialSize;
    const int numApplyGroups =
      (spatialSize + PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD - 1) / (PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD);
    kernels[2].dispatch(
      ctx,
      {input, output, gamma, beta, mask, maskSum, scalar},
      &pc3,
      sizeof(Pass3PC),
      (uint32_t)numApplyGroups,
      (uint32_t)maxBatchSize,
      1u);
  }

  ArrayView<TunableParam> SpatialRMSNormNhwc::sharedParams() {
    static const TunableParam tab[] = {
#define VULKAN_TUNE_PARAM_KERNEL_SPATIAL_RMSNORM_NHWC
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  makeTunableParam<VULKAN_TUNE_PARAM_UNWRAP CANDIDATES>(&VulkanTuneParams::NAME),
#include "vulkantuneparams_fields.inc"
    };
    return ArrayView<TunableParam>(tab);
  }
  ArrayView<TunableParam> SpatialRMSNormNhwc::params() const {
    return sharedParams();
  }

  SpatialRMSNormNhwc::SpatialRMSNormNhwc(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double SpatialRMSNormNhwc::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    const int tile = cfg.spatialRMSNormNhwcTile;
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || tile <= 0)
      return 0.0;
    int chwSize = problemChannels * problemXySize;
    int numSpatialWorkgroups = ceilDivInt(chwSize, tile);
    int numApplyGroups = ceilDivInt(chwSize, PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD);

    double pass1 =
      3.0 * (double)chwSize + (double)numSpatialWorkgroups * (double)(tile - 1);  // mask, square, sum, workgroup reduce
    double pass2 = (double)numSpatialWorkgroups + (double)(tile - 1);
    double pass3Setup =
      (double)numApplyGroups * (double)PASS3_LOCAL_X * 4.0;  // repeated rms calculation per invocation
    double pass3Apply = 4.0 * (double)chwSize;
    return (double)problemBatchSize * (pass1 + pass2 + pass3Setup + pass3Apply);
  }

  bool SpatialRMSNormNhwc::isConfigSupported(int32_t tileSize) {
    return tileSize > 0 && isPow2(tileSize);
  }

  bool SpatialRMSNormNhwc::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    const int tile = cfg.spatialRMSNormNhwcTile;
    if(!isConfigSupported(tile))
      return false;
    if(!workGroupFits((uint32_t)tile, 1u, 1u, lim))
      return false;
    // Shared memory: one array of tile-sized floats (pass1/pass2 partials).
    if((size_t)tile * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

}  // namespace VulkanKernels

#endif  // USE_VULKAN_BACKEND

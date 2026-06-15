#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkankernels.h"
#include <algorithm>
#include <array>
#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/vulkanbackend.h"
#include "vulkanshaders_generated.h"

using std::vector;

// ============================================================================
// TU-local helpers
// ============================================================================

// Power-of-two check shared by the reduction kernels' isConfigSupported()
// predicates (their butterfly reduction requires a power-of-two tile size).
namespace {

  bool isPow2(int32_t v) {
    return v < 0 ? isPow2(-v) : (v > 0 && (v & (v - 1)) == 0);
  }

  template<typename T>
  const T& requireLaunchProfile(const ComputeKernel& kernel) {
    const T* profile = std::get_if<T>(&kernel.launch.data);
    testAssert(profile != nullptr);
    return *profile;
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

  double activationFlopsPerElement(int activation) {
    if(activation == ACTIVATION_IDENTITY)
      return 0.0;
    if(activation == ACTIVATION_RELU)
      return 1.0;
    if(activation == ACTIVATION_SILU)
      return 4.0;
    if(activation == ACTIVATION_MISH || activation == ACTIVATION_MISH_SCALE8)
      return 6.0;
    return 0.0;
  }

  LayerPaddingContract selectedWinogradPaddingContractForEstimate(const VulkanTuneParams& cfg) {
    if(cfg.enableWinogradGemmCoopmat2AccF16 != 0)
      return VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(cfg);
    if(cfg.enableWinogradGemmCoopmat2 != 0)
      return VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(cfg);
    if(cfg.enableWinogradGemmCoopmatAccF16 != 0)
      return VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(cfg);
    if(cfg.enableWinogradGemmCoopmat != 0)
      return VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(cfg);
    if(cfg.enableWinogradGemmDot2AccF16 != 0)
      return VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(cfg);
    if(cfg.enableWinogradGemmDot2 != 0)
      return VulkanKernels::WinogradGemmDot2::layerPaddingContract(cfg);
    return VulkanKernels::WinogradGemm::layerPaddingContract(cfg);
  }

  double
  winogradTransformInvocationCount(int batchSize, int channels, int nnXLen, int nnYLen, const VulkanTuneParams& cfg) {
    if(batchSize <= 0 || channels <= 0 || nnXLen <= 0 || nnYLen <= 0)
      return 0.0;
    LayerPaddingContract contract = selectedWinogradPaddingContractForEstimate(cfg);
    int outTile = winograd3x3OutTileFor(cfg);
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
    return std::string(name) + " tile=[bs=" + std::to_string(blockSize) + " bm=" + std::to_string(bm) +
           " bn=" + std::to_string(bn) + " bk=" + std::to_string(bk) + " wm=" + std::to_string(wm) +
           " wn=" + std::to_string(wn) + " tm=" + std::to_string(tm) + " tn=" + std::to_string(tn) +
           " tk=" + std::to_string(tk) + " warp=" + std::to_string(warp) + "]";
  }

  [[maybe_unused]] std::string
  coopmat2PipelineStatsLabel(std::string_view name, int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
    return std::string(name) + " tile=[bs=" + std::to_string(blockSize) + " bm=" + std::to_string(bm) +
           " bn=" + std::to_string(bn) + " bk=" + std::to_string(bk) + "]";
  }

  [[maybe_unused]] std::string coopmat2PipelineStatsLabel(
    std::string_view name,
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t aligned,
    int32_t packedB,
    std::string_view finalFlagName,
    int32_t finalFlagValue) {
    return std::string(name) + " tile=[bs=" + std::to_string(blockSize) + " bm=" + std::to_string(bm) +
           " bn=" + std::to_string(bn) + " bk=" + std::to_string(bk) + " aligned=" + std::to_string(aligned) +
           " packedB=" + std::to_string(packedB) + " " + std::string(finalFlagName) + "=" +
           std::to_string(finalFlagValue) + "]";
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

namespace VulkanKernels {

  std::vector<VkSpecializationMapEntry> makeSpecMap(int n) {
    std::vector<VkSpecializationMapEntry> entries(n);
    for(int i = 0; i < n; i++) {
      entries[i].constantID = i;
      entries[i].offset = i * sizeof(uint32_t);
      entries[i].size = sizeof(uint32_t);
    }
    return entries;
  }

  std::vector<uint32_t> makeSpecData(std::initializer_list<uint32_t> vals) {
    return std::vector<uint32_t>(vals);
  }

}  // namespace VulkanKernels

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

// VulkanKernels — per-kernel build()/dispatch() bodies
// ============================================================================
// VulkanKernels implementations
// ============================================================================

namespace VulkanKernels {

  // -------- ScaleBiasMaskAct --------

  ComputeKernel ScaleBiasMaskAct::build(VkDevice device, VkPipelineCache cache, bool fp16, int activation) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)activation, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      scale_bias_mask_act_nchw,
      scale_bias_mask_act_nchw_size,
      fp16 ? 10 : 5,
      sizeof(PC),
      specMap,
      specDat,
      cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.debugName = "ScaleBiasMaskAct";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void ScaleBiasMaskAct::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* scale,
    VulkanBuffer* bias,
    VulkanBuffer* mask,
    const PC& pc) {
    kernel.dispatch(
      ctx,
      {input, output, scale, bias, mask},
      &pc,
      sizeof(PC),
      divUpU32(pc.paddedSpatialSize, ScaleBiasMaskAct::LOCAL_SIZE_X),
      (uint32_t)pc.numChannels,
      1u);
  }

  // -------- WinogradTransform --------

  ComputeKernel WinogradTransform::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int inTile,
    int outTile,
    int filterSize,
    int offset,
    int localSizeX,
    int localSizeY,
    int packedBM,
    int packedBK,
    int packedAPadWords) {
    requireSupportedWinogradGeometry("WinogradTransform", filterSize, inTile, outTile);
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
      device, winograd_transform_nchw, winograd_transform_nchw_size, 4, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 0, 1};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.debugName = "WinogradTransform";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradTransform::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    const PC& pc) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    // Writes row-major packed A tiles ([winoTile, M-tile, K-block, M, K/4+pad]).
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

  ArrayView<TunableParam> WinogradTransform::sharedParams() {
    static constexpr int32_t wgVals[] = {4, 8, 16, 32};
    static const std::array<TunableParam, 2> tab = {{
      {&VulkanTuneParams::winogradTransformLocalSizeX, wgVals},
      {&VulkanTuneParams::winogradTransformLocalSizeY, wgVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> WinogradTransform::params() const {
    return sharedParams();
  }

  WinogradTransform::WinogradTransform(int batchSize, int inChannels, int nnXLen, int nnYLen)
    : problemBatchSize(batchSize), problemInChannels(inChannels), problemNnXLen(nnXLen), problemNnYLen(nnYLen) {}

  double WinogradTransform::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    // Approximate transform arithmetic per dispatched (tile, channel) invocation.
    int outTile = winograd3x3OutTileFor(cfg);
    double flopsPerInvocation = (outTile == 2) ? 32.0 : 336.0;
    return winogradTransformInvocationCount(problemBatchSize, problemInChannels, problemNnXLen, problemNnYLen, cfg) *
           flopsPerInvocation;
  }

  bool WinogradTransform::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  bool WinogradTransform::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.winogradTransformLocalSizeX, cfg.winogradTransformLocalSizeY))
      return false;
    if(!workGroupFits((uint32_t)cfg.winogradTransformLocalSizeX, (uint32_t)cfg.winogradTransformLocalSizeY, 1u, lim))
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- WinogradBNActTransform --------

  ComputeKernel WinogradBNActTransform::build(
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
    int packedBM,
    int packedBK,
    int packedAPadWords) {
    requireSupportedWinogradGeometry("WinogradBNActTransform", filterSize, inTile, outTile);
    testAssert(isConfigSupported(localSizeX, localSizeY));
    testAssert(packedBM > 0 && packedBK > 0 && packedBK % 4 == 0 && packedAPadWords >= 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(15);
    auto specDat = makeSpecData(
      {(uint32_t)inTile,
       (uint32_t)inTile,
       (uint32_t)outTile,
       (uint32_t)outTile,
       (uint32_t)filterSize,
       (uint32_t)filterSize,
       (uint32_t)offset,
       (uint32_t)offset,
       (uint32_t)activation,
       fp16 ? 1u : 0u,
       (uint32_t)localSizeX,
       (uint32_t)localSizeY,
       (uint32_t)packedBM,
       (uint32_t)packedBK,
       (uint32_t)packedAPadWords});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      winograd_bn_act_transform_nchw,
      winograd_bn_act_transform_nchw_size,
      fp16 ? 10 : 5,
      sizeof(PC),
      specMap,
      specDat,
      cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.debugName = "WinogradBNActTransform";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradBNActTransform::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* scale,
    VulkanBuffer* bias,
    VulkanBuffer* mask,
    const PC& pc) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    // Writes row-major packed A tiles ([winoTile, M-tile, K-block, M, K/4+pad]).
    testAssert(pc.numInChannelsPadded % 4 == 0);
    testAssert(pc.numTilesReal <= pc.numTilesPadded);
    kernel.dispatch(
      ctx,
      {input, output, scale, bias, mask},
      &pc,
      sizeof(PC),
      divUpU32(pc.numInChannelsPadded, kernel.localSizeX),
      divUpU32(pc.numTilesPadded, kernel.localSizeY),
      1u);
  }

  bool WinogradBNActTransform::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  ArrayView<TunableParam> WinogradBNActTransform::sharedParams() {
    static constexpr int32_t wgVals[] = {4, 8, 16, 32};
    static const std::array<TunableParam, 2> tab = {{
      {&VulkanTuneParams::winogradBNActTransformLocalSizeX, wgVals},
      {&VulkanTuneParams::winogradBNActTransformLocalSizeY, wgVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> WinogradBNActTransform::params() const {
    return sharedParams();
  }

  WinogradBNActTransform::WinogradBNActTransform(int batchSize, int inChannels, int nnXLen, int nnYLen, int activation)
    : problemBatchSize(batchSize),
      problemInChannels(inChannels),
      problemNnXLen(nnXLen),
      problemNnYLen(nnYLen),
      problemActivation(activation) {}

  double WinogradBNActTransform::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    int outTile = winograd3x3OutTileFor(cfg);
    int inTile = outTile + 2;
    double paddedInvocations =
      winogradTransformInvocationCount(problemBatchSize, problemInChannels, problemNnXLen, problemNnYLen, cfg);
    int numTilesX = ceilDivInt(problemNnXLen, outTile);
    int numTilesY = ceilDivInt(problemNnYLen, outTile);
    LayerPaddingContract contract = selectedWinogradPaddingContractForEstimate(cfg);
    int numTilesPadded = roundUpToMultiple(problemBatchSize * numTilesX * numTilesY, std::max(1, contract.m));
    double realChannelInvocations = (double)numTilesPadded * (double)std::max(0, problemInChannels);
    double transformFlopsPerInvocation = (outTile == 2) ? 32.0 : 336.0;
    double prepFlopsPerInput =
      3.0 + activationFlopsPerElement(problemActivation);  // scale+bias, mask multiply, activation
    return paddedInvocations * transformFlopsPerInvocation +
           realChannelInvocations * (double)(inTile * inTile) * prepFlopsPerInput;
  }

  bool WinogradBNActTransform::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.winogradBNActTransformLocalSizeX, cfg.winogradBNActTransformLocalSizeY))
      return false;
    if(!workGroupFits(
         (uint32_t)cfg.winogradBNActTransformLocalSizeX, (uint32_t)cfg.winogradBNActTransformLocalSizeY, 1u, lim))
      return false;
    return true;
  }

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

  // -------- AddChannelBiases --------

  ComputeKernel AddChannelBiases::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, add_channel_biases_nchw, add_channel_biases_nchw_size, fp16 ? 3 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "AddChannelBiases";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void AddChannelBiases::dispatch(
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
      divUpU32(pc.paddedSpatialSize, AddChannelBiases::LOCAL_SIZE_X),
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

  // -------- ExtractChannel0 --------

  ComputeKernel ExtractChannel0::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, extract_channel0_nchw, extract_channel0_nchw_size, fp16 ? 4 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0, 1} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "ExtractChannel0";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void ExtractChannel0::dispatch(
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
      divUpU32(pc.paddedSpatialSize, ExtractChannel0::LOCAL_SIZE_X),
      (uint32_t)maxBatchSize,
      1u);
  }

  // -------- SumChannels --------

  ComputeKernel SumChannels::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, sum_channels_nchw, sum_channels_nchw_size, fp16 ? 3 : 2, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{0, 1};
    kernel.debugName = "SumChannels";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void SumChannels::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(ctx, {mask, maskSum}, &pc, sizeof(PC), 1u, (uint32_t)maxBatchSize, 1u);
  }

  // -------- WinogradUntransform --------

  ComputeKernel WinogradUntransform::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int inTile,
    int outTile,
    int filterSize,
    int localSizeX,
    int localSizeY,
    bool addToOutput) {
    requireSupportedWinogradGeometry("WinogradUntransform", filterSize, inTile, outTile);
    testAssert(isConfigSupported(localSizeX, localSizeY));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(10);
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
       addToOutput ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, winograd_untransform_nchw, winograd_untransform_nchw_size, 4, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 0, 1};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.debugName = "WinogradUntransform";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void WinogradUntransform::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* output,
    const PC& pc,
    int maxBatchSize) {
    testAssert(kernel.localSizeX > 0 && kernel.localSizeY > 0);
    kernel.dispatch(
      ctx,
      {input, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.numTilesX, kernel.localSizeX),
      divUpU32(pc.numTilesY, kernel.localSizeY),
      (uint32_t)(maxBatchSize * pc.numOutChannels));
  }

  bool WinogradUntransform::isConfigSupported(int32_t localSizeX, int32_t localSizeY) {
    return localSizeX > 0 && localSizeY > 0;
  }

  ArrayView<TunableParam> WinogradUntransform::sharedParams() {
    static constexpr int32_t wgVals[] = {4, 8, 16, 32};
    static const std::array<TunableParam, 2> tab = {{
      {&VulkanTuneParams::winogradUntransformLocalSizeX, wgVals},
      {&VulkanTuneParams::winogradUntransformLocalSizeY, wgVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> WinogradUntransform::params() const {
    return sharedParams();
  }

  WinogradUntransform::WinogradUntransform(int batchSize, int outChannels, int nnXLen, int nnYLen)
    : problemBatchSize(batchSize), problemOutChannels(outChannels), problemNnXLen(nnXLen), problemNnYLen(nnYLen) {}

  double WinogradUntransform::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    if(problemBatchSize <= 0 || problemOutChannels <= 0 || problemNnXLen <= 0 || problemNnYLen <= 0)
      return 0.0;
    int outTile = winograd3x3OutTileFor(cfg);
    int numTilesX = ceilDivInt(problemNnXLen, outTile);
    int numTilesY = ceilDivInt(problemNnYLen, outTile);
    double invocations = (double)problemBatchSize * (double)problemOutChannels * (double)numTilesX * (double)numTilesY;
    double flopsPerInvocation = (outTile == 2) ? 24.0 : 150.0;
    return invocations * flopsPerInvocation;
  }

  bool WinogradUntransform::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.winogradUntransformLocalSizeX, cfg.winogradUntransformLocalSizeY))
      return false;
    if(!workGroupFits(
         (uint32_t)cfg.winogradUntransformLocalSizeX, (uint32_t)cfg.winogradUntransformLocalSizeY, 1u, lim))
      return false;
    return true;
  }

  // -------- WinogradGemm --------

  ComputeKernel WinogradGemm::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int32_t tileM,
    int32_t tileN,
    int32_t tileK,
    int32_t rn,
    int32_t k) {
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
    static constexpr int32_t mnVals[] = {32, 64, 128, 256};
    static constexpr int32_t kVals[] = {8, 16, 32, 64};
    static constexpr int32_t rnVals[] = {4, 8};
    static const std::array<TunableParam, 4> tab = {{
      {&VulkanTuneParams::winogradGemmM, mnVals},
      {&VulkanTuneParams::winogradGemmN, mnVals},
      {&VulkanTuneParams::winogradGemmK, kVals},
      {&VulkanTuneParams::winogradGemmRN, rnVals},
    }};
    return tab;
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

  // -------- GemmStridedTiled --------

  ComputeKernel GemmStridedTiled::build(
    VkDevice device,
    VkPipelineCache cache,
    bool fp16,
    int32_t localSizeX,
    int32_t localSizeY,
    int32_t tileK,
    int32_t rn,
    int32_t k,
    bool addToOutput) {
    // The tiled grid has no split-K dimension and gives each C tile a single workgroup,
    // which is required by ADD_TO_OUTPUT.
    testAssert(isConfigSupported(localSizeX, localSizeY, tileK, rn));
    testAssert(k > 0);
    using namespace VulkanShaders;
    // Spec constant order: WG_X, WG_Y, TILE_K, RN, ADD_TO_OUTPUT, USE_FP16_STORAGE, K_SPEC.
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
      device, gemm_strided_tiled, gemm_strided_tiled_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    // Logical buffers are always {A, B, C}; FP16 aliases them onto bindings 3..5.
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1, 2} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)localSizeX;
    kernel.localSizeY = (uint16_t)localSizeY;
    kernel.launch.data = LaunchProfile::GemmRegN{(uint16_t)rn, (uint32_t)k};
    kernel.debugName = "GemmStridedTiled";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void GemmStridedTiled::dispatch(
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
    // The tiled shader uses 8-wide A loads and vec4 B/C paths:
    //   M÷8 (A load stride uses M/8), N÷8 (host padded, B stride still N/4 vec4 units).
    //   K is unconstrained (scalar k-loop). N is padded; C-store guards on N_real ≤ N.
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && pc.N_real <= pc.N);
    // Each workgroup covers (localSizeX*4) m-values and (localSizeY*regN) n-values.
    kernel.dispatch(
      ctx,
      {input, filter, output},
      &pc,
      sizeof(PC),
      divUpU32(pc.M, (int)kernel.localSizeX * 4),
      divUpU32(pc.N, (int)kernel.localSizeY * (int)launch.regN),
      (uint32_t)numBatches);
  }

  ArrayView<TunableParam> GemmStridedTiled::sharedParams() {
    static constexpr int32_t wgVals[] = {4, 8, 16, 32, 64};
    static constexpr int32_t kVals[] = {4, 8, 16, 32};
    static constexpr int32_t rnVals[] = {2, 4, 8};
    static const std::array<TunableParam, 4> tab = {{
      {&VulkanTuneParams::gemmStridedTiledLocalSizeX, wgVals},
      {&VulkanTuneParams::gemmStridedTiledLocalSizeY, wgVals},
      {&VulkanTuneParams::gemmStridedTiledTileK, kVals},
      {&VulkanTuneParams::gemmStridedTiledRN, rnVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> GemmStridedTiled::params() const {
    return sharedParams();
  }

  GemmStridedTiled::GemmStridedTiled(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedTiled::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  bool GemmStridedTiled::isConfigSupported(int32_t localSizeX, int32_t localSizeY, int32_t tileK, int32_t rn) {
    // localSizeX must be even so MM=localSizeX*4 is divisible by 8 for paired A loads.
    // NN = localSizeY*rn must be ÷4: the B-tile does NN/4 vec4 loads (columns).
    return localSizeX > 0 && (localSizeX % 2 == 0) && localSizeY > 0 && tileK > 0 && rn > 0 &&
           ((localSizeY * rn) % 4 == 0);
  }

  bool GemmStridedTiled::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.gemmStridedTiledLocalSizeX,
         cfg.gemmStridedTiledLocalSizeY,
         cfg.gemmStridedTiledTileK,
         cfg.gemmStridedTiledRN))
      return false;
    if(!workGroupFits((uint32_t)cfg.gemmStridedTiledLocalSizeX, (uint32_t)cfg.gemmStridedTiledLocalSizeY, 1u, lim))
      return false;
    // Shared memory:
    //   tileA[TILE_K * (WG_X + 1)] vec4  +  tileB[TILE_K * (WG_Y*RN + 1)] float
    // tileA uses vec4 LDS with +1 vec4 row padding; tileB remains scalar.
    size_t sharedBytes = (size_t)cfg.gemmStridedTiledTileK *
                         (((size_t)cfg.gemmStridedTiledLocalSizeX + 1) * 4 +
                          ((size_t)cfg.gemmStridedTiledLocalSizeY * cfg.gemmStridedTiledRN + 1)) *
                         sizeof(float);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- WinogradGemmDot2 --------

  ComputeKernel WinogradGemmDot2::build(
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
    int32_t k) {
    testAssert(isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp));
    testAssert(k > 0);
    testAssert(k % DOT2_BK == 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(10);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)wmIter,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)warp,
       (uint32_t)k});
    ComputeKernel kernel =
      ComputeKernel::build(device, winograd_gemm_dot2, winograd_gemm_dot2_size, 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmDot2";
    kernel.debugFp16 = true;
    return kernel;
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
    testAssert(k > 0);
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && k % DOT2_BK == 0);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
    testAssert(tile.bm > 0 && tile.bn > 0);
    testAssert(pc.M % tile.bm == 0 && pc.N % tile.bn == 0);
    testAssert(tile.kSpec == 0 || k == (int)tile.kSpec);
    // gx = ceil(M/BM), gy = ceil(N/BN), gz = numBatches
    kernel.dispatch(
      ctx, {A, B, C}, &pc, sizeof(PC), divUpU32(pc.M, tile.bm), divUpU32(pc.N, tile.bn), (uint32_t)numBatches);
  }

  bool WinogradGemmDot2::isConfigSupported(
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t wm,
    int32_t wn,
    int32_t wmIter,
    int32_t tm,
    int32_t tn,
    int32_t warp) {
    if(blockSize <= 0 || bm <= 0 || bn <= 0 || wm <= 0 || wn <= 0)
      return false;
    if(wmIter <= 0 || tm <= 0 || tn <= 0 || warp <= 0)
      return false;
    if(bm % 8 != 0 || bn % 8 != 0)
      return false;
    if(tm % 2 != 0)
      return false;  // TM/2 used in accumulator indexing
    if(bm % wm != 0 || bn % wn != 0)
      return false;  // warp grid tiles the BM×BN block exactly
    if(wm % wmIter != 0)
      return false;
    const int32_t wSubM = wm / wmIter;
    if(wSubM <= 0 || wSubM % tm != 0)
      return false;  // tiwr uses (WSUBM/TM) as a modulo divisor
    const int64_t wnIterNumer = (int64_t)wm * wn;
    const int64_t wnIterDenom = (int64_t)warp * tm * tn * wmIter;
    if(wnIterNumer % wnIterDenom != 0)
      return false;
    const int64_t wnIter64 = wnIterNumer / wnIterDenom;
    if(wnIter64 <= 0 || wnIter64 > INT32_MAX)
      return false;
    const int32_t wnIter = (int32_t)wnIter64;
    if(wn % wnIter != 0)
      return false;
    const int32_t wSubN = wn / wnIter;
    if(wSubN <= 0 || wSubN % tn != 0)
      return false;
    // Every WM×WN warp region in the BM×BN tile needs its own warp: warp_c is
    // computed as warp_i/(BM/WM), so too few warps leaves the right N-columns
    // unwritten. Require exactly one warp per region.
    if((int64_t)blockSize != (int64_t)(bm / wm) * (bn / wn) * warp)
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
         cfg.dot2BlockSize,
         cfg.dot2BM,
         cfg.dot2BN,
         cfg.dot2WM,
         cfg.dot2WN,
         cfg.dot2WMIter,
         cfg.dot2TM,
         cfg.dot2TN,
         cfg.dot2Warp))
      return false;
    // Shared tiles: buf_a[BM*(BK+4)] + buf_b[BN*(BK+4)] scalar float16_t (2 bytes).
    // BK is fixed in the shader (#define BK matches VulkanKernels::DOT2_BK); +4 is the bank-conflict pad.
    size_t sharedBytes = (size_t)(cfg.dot2BM + cfg.dot2BN) * (DOT2_BK + 4) * sizeof(uint16_t);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.dot2BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmDot2AccF16 --------

  ComputeKernel WinogradGemmDot2AccF16::build(
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
    int32_t k) {
    testAssert(isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp));
    testAssert(k > 0);
    testAssert(k % DOT2_BK == 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(10);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)wmIter,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)warp,
       (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device, winograd_gemm_dot2_accf16, winograd_gemm_dot2_accf16_size, 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmDot2AccF16";
    kernel.debugFp16 = true;
    return kernel;
  }

  bool WinogradGemmDot2AccF16::isConfigSupported(
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t wm,
    int32_t wn,
    int32_t wmIter,
    int32_t tm,
    int32_t tn,
    int32_t warp) {
    return WinogradGemmDot2::isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp);
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
         cfg.dot2AccF16BlockSize,
         cfg.dot2AccF16BM,
         cfg.dot2AccF16BN,
         cfg.dot2AccF16WM,
         cfg.dot2AccF16WN,
         cfg.dot2AccF16WMIter,
         cfg.dot2AccF16TM,
         cfg.dot2AccF16TN,
         cfg.dot2AccF16Warp))
      return false;
    size_t sharedBytes = (size_t)(cfg.dot2AccF16BM + cfg.dot2AccF16BN) * (DOT2_BK + 4) * sizeof(uint16_t);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.dot2AccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedDot2 --------

  ComputeKernel GemmStridedDot2::build(
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
    bool addToOutput) {
    // isConfigSupported also guarantees the warp-tile partition is single-writer
    // for C, which is required by ADD_TO_OUTPUT.
    testAssert(WinogradGemmDot2::isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp));
    testAssert(k > 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(14);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)wmIter,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)warp,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    ComputeKernel kernel =
      ComputeKernel::build(device, gemm_strided_dot2, gemm_strided_dot2_size, 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedDot2";
    kernel.debugFp16 = true;
    return kernel;
  }

  void GemmStridedDot2::dispatch(
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
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && pc.N_real <= pc.N);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
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

  bool GemmStridedDot2::isConfigSupported(
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t wm,
    int32_t wn,
    int32_t wmIter,
    int32_t tm,
    int32_t tn,
    int32_t warp) {
    return WinogradGemmDot2::isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp);
  }

  GemmStridedDot2::GemmStridedDot2(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedDot2::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedDot2::sharedParams() {
    return WinogradGemmDot2::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedDot2::params() const {
    return sharedParams();
  }

  bool GemmStridedDot2::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedDot2BlockSize,
         cfg.stridedDot2BM,
         cfg.stridedDot2BN,
         cfg.stridedDot2WM,
         cfg.stridedDot2WN,
         cfg.stridedDot2WMIter,
         cfg.stridedDot2TM,
         cfg.stridedDot2TN,
         cfg.stridedDot2Warp))
      return false;
    size_t sharedBytes = (size_t)(cfg.stridedDot2BM + cfg.stridedDot2BN) * (DOT2_BK + 4) * sizeof(uint16_t);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedDot2BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedDot2AccF16 --------

  ComputeKernel GemmStridedDot2AccF16::build(
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
    bool addToOutput) {
    testAssert(isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp));
    testAssert(k > 0);
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(14);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)wmIter,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)warp,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    ComputeKernel kernel = ComputeKernel::build(
      device, gemm_strided_dot2_accf16, gemm_strided_dot2_accf16_size, 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedDot2AccF16";
    kernel.debugFp16 = true;
    return kernel;
  }

  bool GemmStridedDot2AccF16::isConfigSupported(
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t wm,
    int32_t wn,
    int32_t wmIter,
    int32_t tm,
    int32_t tn,
    int32_t warp) {
    return GemmStridedDot2::isConfigSupported(blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp);
  }

  void GemmStridedDot2AccF16::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* filter,
    VulkanBuffer* output,
    int k,
    const PC& pc,
    int numBatches) {
    GemmStridedDot2::dispatch(ctx, kernel, input, filter, output, k, pc, numBatches);
  }

  GemmStridedDot2AccF16::GemmStridedDot2AccF16(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedDot2AccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedDot2AccF16::sharedParams() {
    return GemmStridedDot2::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedDot2AccF16::params() const {
    return sharedParams();
  }

  bool GemmStridedDot2AccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedDot2AccF16BlockSize,
         cfg.stridedDot2AccF16BM,
         cfg.stridedDot2AccF16BN,
         cfg.stridedDot2AccF16WM,
         cfg.stridedDot2AccF16WN,
         cfg.stridedDot2AccF16WMIter,
         cfg.stridedDot2AccF16TM,
         cfg.stridedDot2AccF16TN,
         cfg.stridedDot2AccF16Warp))
      return false;
    size_t sharedBytes = (size_t)(cfg.stridedDot2AccF16BM + cfg.stridedDot2AccF16BN) * (DOT2_BK + 4) * sizeof(uint16_t);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedDot2AccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- Coopmat (VK_KHR_cooperative_matrix) shared geometry check --------
  //
  // Unlike DOT2, the coopmat inner loop has no WMITER/register-pair packing:
  // each warp owns a (WM/TM)x(WN/TN) grid of TMxTN accumulator fragments and
  // contracts BK in TK steps. TM/TN/TK must match a device-reported shape (that
  // membership check lives in bench(), which has access to the device shape list).
  static bool coopmatConfigSupported(
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
    if(blockSize <= 0 || bm <= 0 || bn <= 0 || bk <= 0 || wm <= 0 || wn <= 0)
      return false;
    if(tm <= 0 || tn <= 0 || tk <= 0 || warp <= 0)
      return false;
    if(bm % 8 != 0 || bn % 8 != 0)
      return false;  // row-major packed A and pack4-N B cooperative loads
    if(bk % 4 != 0)
      return false;  // K always ÷4
    if(bm % wm != 0 || bn % wn != 0)
      return false;  // warp grid tiles the BM×BN block exactly
    if(wm % tm != 0 || wn % tn != 0)
      return false;  // fragment grid tiles the WM×WN warp tile exactly
    if(bk % tk != 0)
      return false;  // K contracted in TK steps over the BK slab
    // One warp (subgroup) per WM×WN region: warp_c = warp_i/(BM/WM).
    if((int64_t)blockSize != (int64_t)(bm / wm) * (bn / wn) * warp)
      return false;
    return true;
  }

  constexpr int32_t STRIDED_COOPMAT_A_PAD_V8 = 2;

  // Strided coopmat staging: buf_a[BK*(BM/8+A_PAD_V8)] uvec4 plus
  // buf_b[BN*(BK+8)] f16, plus coopStage[NUM_WARPS*TM*TN] fp32 for the
  // guarded store. Returns bytes.
  static size_t coopmatSharedBytes(
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t blockSize,
    int32_t wm,
    int32_t wn,
    int32_t warp,
    int32_t tm,
    int32_t tn) {
    const int32_t numWarps = (wm > 0 && wn > 0 && warp > 0) ? blockSize / warp : 1;
    size_t aBytes = (size_t)bk * ((size_t)bm / 8 + STRIDED_COOPMAT_A_PAD_V8) * sizeof(uint32_t) * 4;
    size_t bBytes = (size_t)bn * ((size_t)bk + STRIDED_COOPMAT_PACKED_B_PAD_SCALARS) * sizeof(uint16_t);
    size_t stageBytes = (size_t)std::max(1, numWarps) * tm * tn * sizeof(float);
    return aBytes + bBytes + stageBytes;
  }

  static size_t coopmatAccF16SharedBytes(
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t blockSize,
    int32_t wm,
    int32_t wn,
    int32_t warp,
    int32_t tm,
    int32_t tn) {
    const int32_t numWarps = (wm > 0 && wn > 0 && warp > 0) ? blockSize / warp : 1;
    size_t aBytes = (size_t)bk * ((size_t)bm / 8 + STRIDED_COOPMAT_A_PAD_V8) * sizeof(uint32_t) * 4;
    size_t bBytes = (size_t)bn * ((size_t)bk + STRIDED_COOPMAT_PACKED_B_PAD_SCALARS) * sizeof(uint16_t);
    size_t stageBytes = (size_t)std::max(1, numWarps) * tm * tn * sizeof(uint16_t);
    return aBytes + bBytes + stageBytes;
  }

  // Winograd coopmat uses packed half8 shared staging over even half4 strides:
  // buf_a[BM*(BK/4+2)/2] + buf_b[BK*(BN/4+2)/2] uvec4.
  static size_t winogradCoopmatSharedBytes(int32_t bm, int32_t bn, int32_t bk) {
    constexpr size_t packedPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    size_t aBytes = (size_t)bm * ((size_t)bk / 4 + packedPadWords) * sizeof(uint32_t) * 2;
    size_t bBytes = (size_t)bk * ((size_t)bn / 4 + packedPadWords) * sizeof(uint32_t) * 2;
    return aBytes + bBytes;
  }

  static size_t winogradCoopmatAccF16SharedBytes(int32_t bm, int32_t bn, int32_t bk) {
    constexpr size_t packedPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    size_t aBytes = (size_t)bm * ((size_t)bk / 4 + packedPadWords) * sizeof(uint32_t) * 2;
    size_t bBytes = (size_t)bk * ((size_t)bn / 4 + packedPadWords) * sizeof(uint32_t) * 2;
    return aBytes + bBytes;
  }

  // -------- WinogradGemmCoopmat --------

  ComputeKernel WinogradGemmCoopmat::build(
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
    uint32_t requiredSubgroupSize) {
    testAssert(isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    // WARP (the shader's per-subgroup lane count spec constant) must equal the
    // pipeline's required subgroup size; coopmat fragments are subgroup-scoped,
    // so a mismatch would map lanes to the wrong fragment elements.
    testAssert((uint32_t)warp == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap(11);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)warp,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmatPipelineStatsLabel("WinogradGemmCoopmat", blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      winograd_gemm_coopmat,
      winograd_gemm_coopmat_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmCoopmat";
    kernel.debugFp16 = true;
    return kernel;
#else
    // Coopmat shaders were not built by this toolchain; supportsCoopmat1F16 is
    // forced false at detection, so this path is never selected at runtime.
    (void)device;
    (void)cache;
    testAssert(false && "WinogradGemmCoopmat::build called but coopmat shaders were not compiled");
    return ComputeKernel();
#endif
  }

  void WinogradGemmCoopmat::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* A,
    VulkanBuffer* B,
    VulkanBuffer* C,
    int k,
    const PC& pc,
    int numBatches) {
    testAssert(k > 0);
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && k % 8 == 0);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
    testAssert(tile.bm > 0 && tile.bn > 0);
    testAssert(pc.M % tile.bm == 0 && pc.N % tile.bn == 0);
    testAssert(tile.kSpec == 0 || k == (int)tile.kSpec);
    kernel.dispatch(
      ctx, {A, B, C}, &pc, sizeof(PC), divUpU32(pc.M, tile.bm), divUpU32(pc.N, tile.bn), (uint32_t)numBatches);
  }

  bool WinogradGemmCoopmat::isConfigSupported(
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
    if(tm % 4 != 0 || tn % 8 != 0 || tk % 8 != 0)
      return false;  // packed half8 coopMatLoad offsets must be fragment-aligned
    if(bk % 8 != 0)
      return false;  // A is transported as packed half8 rows before shared staging
    return coopmatConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
  }

  WinogradGemmCoopmat::WinogradGemmCoopmat(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmat::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmat::sharedParams() {
    static const std::array<TunableParam, 0> tab = {{}};
    return tab;
  }

  ArrayView<TunableParam> WinogradGemmCoopmat::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmat::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.coopmatBlockSize,
         cfg.coopmatBM,
         cfg.coopmatBN,
         cfg.coopmatBK,
         cfg.coopmatWM,
         cfg.coopmatWN,
         cfg.coopmatTM,
         cfg.coopmatTN,
         cfg.coopmatTK,
         cfg.coopmatWarp))
      return false;
    size_t sharedBytes = winogradCoopmatSharedBytes(cfg.coopmatBM, cfg.coopmatBN, cfg.coopmatBK);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmatBlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat --------

  ComputeKernel GemmStridedCoopmat::build(
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
    bool addToOutput) {
    // isConfigSupported also guarantees the coopmat tile partition is single-writer
    // for C, which is required by ADD_TO_OUTPUT.
    testAssert(isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp));
    testAssert(k > 0);
    // WARP (the shader's per-subgroup lane count spec constant) must equal the
    // pipeline's required subgroup size; coopmat fragments are subgroup-scoped,
    // so a mismatch would map lanes to the wrong fragment elements.
    testAssert((uint32_t)warp == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap(15);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)warp,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmatPipelineStatsLabel("GemmStridedCoopmat", blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      gemm_strided_coopmat,
      gemm_strided_coopmat_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedCoopmat";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)packedB;
    (void)kAligned;
    (void)addToOutput;
    testAssert(false && "GemmStridedCoopmat::build called but coopmat shaders were not compiled");
    return ComputeKernel();
#endif
  }

  void GemmStridedCoopmat::dispatch(
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
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && pc.N_real <= pc.N);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
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

  bool GemmStridedCoopmat::isConfigSupported(
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
    if(!coopmatConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp))
      return false;
    if(tm % 8 != 0 || wm % 8 != 0)
      return false;  // packed half8 A coopMatLoad offsets must start on M/8 boundaries
    return true;
  }

  GemmStridedCoopmat::GemmStridedCoopmat(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedCoopmat::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedCoopmat::sharedParams() {
    return WinogradGemmCoopmat::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedCoopmat::params() const {
    return sharedParams();
  }

  bool GemmStridedCoopmat::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedCoopmatBlockSize,
         cfg.stridedCoopmatBM,
         cfg.stridedCoopmatBN,
         cfg.stridedCoopmatBK,
         cfg.stridedCoopmatWM,
         cfg.stridedCoopmatWN,
         cfg.stridedCoopmatTM,
         cfg.stridedCoopmatTN,
         cfg.stridedCoopmatTK,
         cfg.stridedCoopmatWarp))
      return false;
    size_t sharedBytes = coopmatSharedBytes(
      cfg.stridedCoopmatBM,
      cfg.stridedCoopmatBN,
      cfg.stridedCoopmatBK,
      cfg.stridedCoopmatBlockSize,
      cfg.stridedCoopmatWM,
      cfg.stridedCoopmatWN,
      cfg.stridedCoopmatWarp,
      cfg.stridedCoopmatTM,
      cfg.stridedCoopmatTN);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedCoopmatBlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmCoopmatAccF16 --------

  ComputeKernel WinogradGemmCoopmatAccF16::build(
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
    uint32_t requiredSubgroupSize) {
    testAssert(isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    testAssert((uint32_t)warp == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap(11);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)warp,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel =
        coopmatPipelineStatsLabel("WinogradGemmCoopmatAccF16", blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      winograd_gemm_coopmat_accf16,
      winograd_gemm_coopmat_accf16_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmCoopmatAccF16";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    testAssert(false && "WinogradGemmCoopmatAccF16::build called but coopmat shaders were not compiled");
    return ComputeKernel();
#endif
  }

  WinogradGemmCoopmatAccF16::WinogradGemmCoopmatAccF16(int m, int n, int k, int numBatches)
    : problemM(m), problemN(n), problemK(k), problemNumBatches(numBatches) {}

  double WinogradGemmCoopmatAccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemNumBatches, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> WinogradGemmCoopmatAccF16::sharedParams() {
    return WinogradGemmCoopmat::sharedParams();
  }

  ArrayView<TunableParam> WinogradGemmCoopmatAccF16::params() const {
    return sharedParams();
  }

  bool WinogradGemmCoopmatAccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.coopmatAccF16BlockSize,
         cfg.coopmatAccF16BM,
         cfg.coopmatAccF16BN,
         cfg.coopmatAccF16BK,
         cfg.coopmatAccF16WM,
         cfg.coopmatAccF16WN,
         cfg.coopmatAccF16TM,
         cfg.coopmatAccF16TN,
         cfg.coopmatAccF16TK,
         cfg.coopmatAccF16Warp))
      return false;
    size_t sharedBytes =
      winogradCoopmatAccF16SharedBytes(cfg.coopmatAccF16BM, cfg.coopmatAccF16BN, cfg.coopmatAccF16BK);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmatAccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmatAccF16 --------

  ComputeKernel GemmStridedCoopmatAccF16::build(
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
    bool addToOutput) {
    testAssert(isConfigSupported(blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp));
    testAssert(k > 0);
    testAssert((uint32_t)warp == requiredSubgroupSize);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    auto specMap = makeSpecMap(15);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)wm,
       (uint32_t)wn,
       (uint32_t)tm,
       (uint32_t)tn,
       (uint32_t)tk,
       (uint32_t)warp,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel =
        coopmatPipelineStatsLabel("GemmStridedCoopmatAccF16", blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      gemm_strided_coopmat_accf16,
      gemm_strided_coopmat_accf16_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ true,
      requiredSubgroupSize,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedCoopmatAccF16";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)packedB;
    (void)kAligned;
    (void)addToOutput;
    testAssert(false && "GemmStridedCoopmatAccF16::build called but coopmat shaders were not compiled");
    return ComputeKernel();
#endif
  }

  GemmStridedCoopmatAccF16::GemmStridedCoopmatAccF16(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedCoopmatAccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedCoopmatAccF16::sharedParams() {
    return WinogradGemmCoopmat::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedCoopmatAccF16::params() const {
    return sharedParams();
  }

  bool GemmStridedCoopmatAccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedCoopmatAccF16BlockSize,
         cfg.stridedCoopmatAccF16BM,
         cfg.stridedCoopmatAccF16BN,
         cfg.stridedCoopmatAccF16BK,
         cfg.stridedCoopmatAccF16WM,
         cfg.stridedCoopmatAccF16WN,
         cfg.stridedCoopmatAccF16TM,
         cfg.stridedCoopmatAccF16TN,
         cfg.stridedCoopmatAccF16TK,
         cfg.stridedCoopmatAccF16Warp))
      return false;
    size_t sharedBytes = coopmatAccF16SharedBytes(
      cfg.stridedCoopmatAccF16BM,
      cfg.stridedCoopmatAccF16BN,
      cfg.stridedCoopmatAccF16BK,
      cfg.stridedCoopmatAccF16BlockSize,
      cfg.stridedCoopmatAccF16WM,
      cfg.stridedCoopmatAccF16WN,
      cfg.stridedCoopmatAccF16Warp,
      cfg.stridedCoopmatAccF16TM,
      cfg.stridedCoopmatAccF16TN);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedCoopmatAccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmCoopmat2 --------

  ComputeKernel WinogradGemmCoopmat2::build(
    VkDevice device,
    VkPipelineCache cache,
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t k) {
    static_assert(sizeof(PC) <= 128, "WinogradGemmCoopmat2 push constants must fit Vulkan's guaranteed minimum");
    testAssert(isConfigSupported(blockSize, bm, bn, bk));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap(5);
    auto specDat = makeSpecData({(uint32_t)blockSize, (uint32_t)bm, (uint32_t)bn, (uint32_t)bk, (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmat2PipelineStatsLabel("WinogradGemmCoopmat2", blockSize, bm, bn, bk);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      winograd_gemm_coopmat2,
      winograd_gemm_coopmat2_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ false,
      /*requiredSubgroupSize*/ 0,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmCoopmat2";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    testAssert(false && "WinogradGemmCoopmat2::build called but coopmat2 shaders were not compiled");
    return ComputeKernel();
#endif
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
    testAssert(k > 0);
    testAssert(pc.M % 8 == 0 && pc.N % 8 == 0 && k % 8 == 0);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
    testAssert(tile.bm > 0 && tile.bn > 0);
    testAssert(pc.M % tile.bm == 0 && pc.N % tile.bn == 0);
    testAssert(tile.kSpec == 0 || k == (int)tile.kSpec);
    kernel.dispatch(
      ctx, {A, B, C}, &pc, sizeof(PC), divUpU32(pc.M, tile.bm), divUpU32(pc.N, tile.bn), (uint32_t)numBatches);
  }

  bool WinogradGemmCoopmat2::isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
    if(blockSize <= 0 || bm <= 0 || bn <= 0 || bk <= 0)
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
    if(!isConfigSupported(cfg.coopmat2BlockSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat2BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat2 --------

  ComputeKernel GemmStridedCoopmat2::build(
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
    bool addToOutput) {
    static_assert(sizeof(PC) <= 128, "GemmStridedCoopmat2 push constants must fit Vulkan's guaranteed minimum");
    testAssert(isConfigSupported(blockSize, bm, bn, bk));
    testAssert(k > 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap(9);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmat2PipelineStatsLabel(
        "GemmStridedCoopmat2", blockSize, bm, bn, bk, aligned, packedB, "kAligned", kAligned);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      gemm_strided_coopmat2,
      gemm_strided_coopmat2_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ false,
      /*requiredSubgroupSize*/ 0,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedCoopmat2";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)packedB;
    (void)kAligned;
    (void)addToOutput;
    testAssert(false && "GemmStridedCoopmat2::build called but coopmat2 shaders were not compiled");
    return ComputeKernel();
#endif
  }

  void GemmStridedCoopmat2::dispatch(
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
    testAssert(pc.M % 4 == 0 && pc.N % 4 == 0 && pc.N_real <= pc.N);
    const auto& tile = std::get<LaunchProfile::GemmDot2Tile>(kernel.launch.data);
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

  bool GemmStridedCoopmat2::isConfigSupported(int32_t blockSize, int32_t bm, int32_t bn, int32_t bk) {
    if(blockSize <= 0 || bm <= 0 || bn <= 0 || bk <= 0)
      return false;
    if(bm % 8 != 0 || bn % 8 != 0 || bk % 4 != 0)
      return false;
    return true;
  }

  GemmStridedCoopmat2::GemmStridedCoopmat2(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedCoopmat2::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedCoopmat2::sharedParams() {
    return WinogradGemmCoopmat2::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedCoopmat2::params() const {
    return sharedParams();
  }

  bool GemmStridedCoopmat2::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedCoopmat2BlockSize, cfg.stridedCoopmat2BM, cfg.stridedCoopmat2BN, cfg.stridedCoopmat2BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedCoopmat2BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- WinogradGemmCoopmat2AccF16 --------

  ComputeKernel WinogradGemmCoopmat2AccF16::build(
    VkDevice device,
    VkPipelineCache cache,
    int32_t blockSize,
    int32_t bm,
    int32_t bn,
    int32_t bk,
    int32_t k) {
    static_assert(sizeof(PC) <= 128, "WinogradGemmCoopmat2AccF16 push constants must fit Vulkan's guaranteed minimum");
    testAssert(isConfigSupported(blockSize, bm, bn, bk));
    testAssert(k > 0);
    testAssert(k % bk == 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap(5);
    auto specDat = makeSpecData({(uint32_t)blockSize, (uint32_t)bm, (uint32_t)bn, (uint32_t)bk, (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmat2PipelineStatsLabel("WinogradGemmCoopmat2AccF16", blockSize, bm, bn, bk);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      winograd_gemm_coopmat2_accf16,
      winograd_gemm_coopmat2_accf16_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ false,
      /*requiredSubgroupSize*/ 0,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "WinogradGemmCoopmat2AccF16";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    testAssert(false && "WinogradGemmCoopmat2AccF16::build called but coopmat2 shaders were not compiled");
    return ComputeKernel();
#endif
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
         cfg.coopmat2AccF16BlockSize, cfg.coopmat2AccF16BM, cfg.coopmat2AccF16BN, cfg.coopmat2AccF16BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.coopmat2AccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmStridedCoopmat2AccF16 --------

  ComputeKernel GemmStridedCoopmat2AccF16::build(
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
    bool addToOutput) {
    static_assert(sizeof(PC) <= 128, "GemmStridedCoopmat2AccF16 push constants must fit Vulkan's guaranteed minimum");
    testAssert(isConfigSupported(blockSize, bm, bn, bk));
    testAssert(k > 0);
    using namespace VulkanShaders;
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    auto specMap = makeSpecMap(9);
    auto specDat = makeSpecData(
      {(uint32_t)blockSize,
       (uint32_t)bm,
       (uint32_t)bn,
       (uint32_t)bk,
       (uint32_t)aligned,
       addToOutput ? 1u : 0u,
       (uint32_t)packedB,
       (uint32_t)kAligned,
       (uint32_t)k});
    std::string statsLabel;
    std::string_view statsLabelView;
    if(VulkanHelpers::pipelineExecutableStatsEnabled(device)) {
      statsLabel = coopmat2PipelineStatsLabel(
        "GemmStridedCoopmat2AccF16", blockSize, bm, bn, bk, aligned, packedB, "kAligned", kAligned);
      statsLabelView = statsLabel;
    }
    ComputeKernel kernel = ComputeKernel::build(
      device,
      gemm_strided_coopmat2_accf16,
      gemm_strided_coopmat2_accf16_size,
      3,
      sizeof(PC),
      specMap,
      specDat,
      cache,
      /*requireFullSubgroups*/ false,
      /*requiredSubgroupSize*/ 0,
      statsLabelView);
    kernel.bindingMap = {0, 1, 2};
    kernel.localSizeX = (uint16_t)blockSize;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::GemmDot2Tile{(uint16_t)bm, (uint16_t)bn, (uint32_t)k};
    kernel.debugName = "GemmStridedCoopmat2AccF16";
    kernel.debugFp16 = true;
    return kernel;
#else
    (void)device;
    (void)cache;
    (void)aligned;
    (void)packedB;
    (void)kAligned;
    (void)addToOutput;
    testAssert(false && "GemmStridedCoopmat2AccF16::build called but coopmat2 shaders were not compiled");
    return ComputeKernel();
#endif
  }

  GemmStridedCoopmat2AccF16::GemmStridedCoopmat2AccF16(int batchSize, int M, int N, int K)
    : problemBatchSize(batchSize), problemM(M), problemN(N), problemK(K) {}

  double GemmStridedCoopmat2AccF16::estimatedFlopsPerDispatch(const VulkanTuneParams& /*cfg*/) const {
    return gemmFlopsPerDispatch(problemBatchSize, problemM, problemN, problemK);
  }

  ArrayView<TunableParam> GemmStridedCoopmat2AccF16::sharedParams() {
    return WinogradGemmCoopmat2::sharedParams();
  }

  ArrayView<TunableParam> GemmStridedCoopmat2AccF16::params() const {
    return sharedParams();
  }

  bool GemmStridedCoopmat2AccF16::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(
         cfg.stridedCoopmat2AccF16BlockSize,
         cfg.stridedCoopmat2AccF16BM,
         cfg.stridedCoopmat2AccF16BN,
         cfg.stridedCoopmat2AccF16BK))
      return false;
    if(!workGroupFits((uint32_t)cfg.stridedCoopmat2AccF16BlockSize, 1u, 1u, lim))
      return false;
    return true;
  }

  // bench() defined in vulkantuner.cpp.

  // -------- GemmDirectFP32 --------

  ComputeKernel
  GemmDirectFP32::build(VkDevice device, VkPipelineCache cache, int32_t localSizeX, int32_t localSizeY, int32_t k) {
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
    // gemmDirect is used by FC heads where M is the inference batch size, often
    // much smaller than a subgroup. Try narrow-X/wide-Y shapes so a subgroup can
    // cover one or two batch rows and many output channels instead of idling
    // lanes with m >= M.
    static constexpr int32_t wgXVals[] = {1, 2, 4, 8, 16, 32};
    static constexpr int32_t wgYVals[] = {1, 2, 4, 8, 16, 32};
    static const std::array<TunableParam, 2> tab = {{
      {&VulkanTuneParams::gemmDirectLocalSizeX, wgXVals},
      {&VulkanTuneParams::gemmDirectLocalSizeY, wgYVals},
    }};
    return tab;
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

  // -------- AttentionTiled --------

  ComputeKernel AttentionTiled::build(
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
    bool learnableRope) {
    testAssert(isConfigSupported(blockQ, blockKV, qPerThread));
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
      transformer_attention_tiled,
      transformer_attention_tiled_size,
      useRope ? 12u : (fp16 ? 10u : 5u),
      sizeof(PC),
      specMap,
      specDat,
      cache);
    if(useRope) {
      // logical buffers: {Q, K, V, output, mask, cos, sin}
      // bindings 10/11 are cos/sin; 5-9 are FP16 aliases that are statically dead in FP32 builds.
      kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6}
                               : std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 5, 6};
    } else {
      kernel.bindingMap =
        fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 2, 3, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    }
    kernel.localSizeX = (uint16_t)blockQ;
    kernel.localSizeY = 1;
    kernel.launch.data = LaunchProfile::AttentionTiledQ{(uint16_t)qPerThread, (uint32_t)seqLen};
    kernel.debugName = "AttentionTiled";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void AttentionTiled::dispatch(
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
    testAssert(seqLen > 0);
    testAssert(launch.seqLenSpec == 0 || seqLen == (int)launch.seqLenSpec);
    testAssert(pc.numKVHeads > 0 && pc.numHeads % pc.numKVHeads == 0);
    // localSizeX == ATTN_BLOCK_Q; launch.qPerThread is positions per thread.
    uint32_t qBlockSize = (uint32_t)kernel.localSizeX * (uint32_t)launch.qPerThread;
    uint32_t gx = divUpU32((uint32_t)seqLen, qBlockSize);
    if(kernel.bindingMap.size() == 5 || kernel.bindingMap.size() == 10) {
      kernel.dispatch(ctx, {Q, K, V, output, mask}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    } else {
      testAssert(kernel.bindingMap.size() == 12);
      testAssert(cos != nullptr && sin != nullptr);
      kernel.dispatch(ctx, {Q, K, V, output, mask, cos, sin}, &pc, sizeof(PC), gx, (uint32_t)pc.numBH, 1u);
    }
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
    bool learnableRope)
    : problemBatchSize(batchSize),
      problemNumTokens(numTokens),
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

  ArrayView<TunableParam> AttentionTiled::sharedParams() {
    static constexpr int32_t blockVals[] = {16, 32, 64};
    static constexpr int32_t qptVals[] = {1, 2, 4};
    static const std::array<TunableParam, 3> tab = {{
      {&VulkanTuneParams::attnBlockQ, blockVals},
      {&VulkanTuneParams::attnBlockKV, blockVals},
      {&VulkanTuneParams::attnQPerThread, qptVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> AttentionTiled::params() const {
    return sharedParams();
  }

  bool AttentionTiled::isConfigSupported(int32_t blockQ, int32_t blockKV, int32_t qPerThread) {
    // transformer_attention_tiled.glsl loads kMaskTile with one write per localIdx,
    // so every KV-lane must have a writer in the workgroup.
    return blockQ > 0 && blockKV > 0 && qPerThread > 0 && blockKV <= blockQ;
  }

  bool AttentionTiled::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.attnBlockQ, cfg.attnBlockKV, cfg.attnQPerThread))
      return false;
    if(problemUseRope && (problemHeadDim % 2) != 0)
      return false;
    if(!workGroupFits((uint32_t)cfg.attnBlockQ, 1u, 1u, lim))
      return false;
    size_t sharedBytes = (size_t)cfg.attnBlockKV * ((size_t)problemHeadDim + problemVHeadDim + 1) * sizeof(float);
    if(sharedBytes > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp.

  // -------- SwiGLU --------

  ComputeKernel SwiGLU::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, transformer_swiglu, transformer_swiglu_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1, 2} : std::vector<uint8_t>{0, 1, 2};
    kernel.debugName = "SwiGLU";
    kernel.debugFp16 = fp16;
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
    kernel.dispatch(ctx, {main, gate, output}, &pc, sizeof(PC), divUpU32(numVec4, SwiGLU::LOCAL_SIZE_X), 1u, 1u);
  }

  // -------- GPoolReduction --------

  ComputeKernel GPoolReduction::build(VkDevice device, VkPipelineCache cache, bool fp16, int xyStride) {
    testAssert(isConfigSupported(xyStride));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)xyStride, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, gpool_nchw_mask, gpool_nchw_mask_size, fp16 ? 6 : 4, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 0, 2} : std::vector<uint8_t>{0, 1, 2, 3};
    kernel.debugName = "GPoolReduction";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void GPoolReduction::dispatch(
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

  ArrayView<TunableParam> GPoolReduction::sharedParams() {
    // Power-of-2 only (butterfly reduction).
    static constexpr int32_t tileVals[] = {16, 32, 64, 128};
    static const std::array<TunableParam, 1> tab = {{
      {&VulkanTuneParams::gpoolXystride, tileVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> GPoolReduction::params() const {
    return sharedParams();
  }

  GPoolReduction::GPoolReduction(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double GPoolReduction::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || cfg.gpoolXystride <= 0)
      return 0.0;
    double perChannel = 4.0 * (double)problemXySize + 2.0 * (double)(cfg.gpoolXystride - 1) +
                        5.0;  // sum/max loop, reduction, final scale
    return (double)problemBatchSize * (double)problemChannels * perChannel;
  }

  bool GPoolReduction::isConfigSupported(int32_t xyStride) {
    return xyStride > 0 && isPow2(xyStride);
  }

  bool GPoolReduction::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.gpoolXystride))
      return false;
    if(!workGroupFits((uint32_t)cfg.gpoolXystride, 1u, 1u, lim))
      return false;
    // Shared memory: 2 arrays x gpoolXystride floats (partialSums + partialMaxes).
    if((size_t)cfg.gpoolXystride * 2 * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- ValueHeadPool --------

  ComputeKernel ValueHeadPool::build(VkDevice device, VkPipelineCache cache, bool fp16, int xyStride) {
    testAssert(isConfigSupported(xyStride));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)xyStride, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, value_head_pool_nchw, value_head_pool_nchw_size, fp16 ? 4 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0} : std::vector<uint8_t>{0, 1, 2};
    kernel.debugName = "ValueHeadPool";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void ValueHeadPool::dispatch(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* convOut,
    VulkanBuffer* concat,
    VulkanBuffer* maskSum,
    const PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx, {convOut, concat, maskSum}, &pc, sizeof(PC), 1u, (uint32_t)pc.numChannels, (uint32_t)maxBatchSize);
  }

  ArrayView<TunableParam> ValueHeadPool::sharedParams() {
    static constexpr int32_t tileVals[] = {16, 32, 64, 128};
    static const std::array<TunableParam, 1> tab = {{
      {&VulkanTuneParams::valueHeadPoolXystride, tileVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> ValueHeadPool::params() const {
    return sharedParams();
  }

  ValueHeadPool::ValueHeadPool(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double ValueHeadPool::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || cfg.valueHeadPoolXystride <= 0)
      return 0.0;
    double perChannel =
      (double)problemXySize + (double)(cfg.valueHeadPoolXystride - 1) + 9.0;  // sum loop, reduction, final features
    return (double)problemBatchSize * (double)problemChannels * perChannel;
  }

  bool ValueHeadPool::isConfigSupported(int32_t xyStride) {
    return xyStride > 0 && isPow2(xyStride);
  }

  bool ValueHeadPool::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.valueHeadPoolXystride))
      return false;
    if(!workGroupFits((uint32_t)cfg.valueHeadPoolXystride, 1u, 1u, lim))
      return false;
    // Shared memory: one array of valueHeadPoolXystride floats (partialSums).
    if((size_t)cfg.valueHeadPoolXystride * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

  // -------- TransformerRMSNorm --------

  ComputeKernel TransformerRMSNorm::build(VkDevice device, VkPipelineCache cache, bool fp16, bool useSubgroupVariant) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({8u, 8u, 1u, fp16 ? 1u : 0u});
    const uint32_t* spirvData = useSubgroupVariant ? transformer_rmsnorm_subgroup : transformer_rmsnorm;
    const size_t spirvWords = useSubgroupVariant ? transformer_rmsnorm_subgroup_size : transformer_rmsnorm_size;
    ComputeKernel kernel =
      ComputeKernel::build(device, spirvData, spirvWords, fp16 ? 8 : 5, sizeof(PC), specMap, specDat, cache);
    // logical: {input, output, gamma, beta, mask}; gamma/beta stay FP32 in FP16 mode.
    // Bindings: 0-4 (FP32), 5-7 (FP16 aliases: input,output,mask).
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 0, 1, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4};
    kernel.localSizeX = 8;
    kernel.debugName = useSubgroupVariant ? "TransformerRMSNorm.subgroup" : "TransformerRMSNorm";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void TransformerRMSNorm::dispatch(
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

  // -------- Conv2dDirect --------

  ComputeKernel Conv2dDirect::build(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(4);
    auto specDat = makeSpecData({8u, 4u, 4u, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device, conv2d_direct_nchw, conv2d_direct_nchw_size, fp16 ? 6 : 3, sizeof(PC), specMap, specDat, cache);
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 2, 1} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = 8;
    kernel.localSizeY = 4;
    kernel.debugName = "Conv2dDirect";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void Conv2dDirect::dispatch(
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

  ComputeKernel SpatialRMSNorm::buildPass1(VkDevice device, VkPipelineCache cache, bool fp16, int tileSize) {
    testAssert(isConfigSupported(tileSize));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)tileSize, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_spatial_rmsnorm_pass1,
      transformer_spatial_rmsnorm_pass1_size,
      fp16 ? 5 : 3,
      sizeof(Pass1PC),
      specMap,
      specDat,
      cache);
    // logical: {input, mask, partials}; partials stays FP32 in FP16 mode.
    kernel.bindingMap = fp16 ? std::vector<uint8_t>{0, 1, 2, 0, 1} : std::vector<uint8_t>{0, 1, 2};
    kernel.localSizeX = (uint16_t)tileSize;
    kernel.debugName = "SpatialRMSNorm.pass1";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  ComputeKernel
  SpatialRMSNorm::buildPass2(VkDevice device, VkPipelineCache cache, int tileSize, bool useSubgroupVariant) {
    testAssert(isConfigSupported(tileSize));
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(1);
    auto specDat = makeSpecData({(uint32_t)tileSize});
    const uint32_t* spirvData =
      useSubgroupVariant ? transformer_spatial_rmsnorm_pass2_subgroup : transformer_spatial_rmsnorm_pass2;
    const size_t spirvWords =
      useSubgroupVariant ? transformer_spatial_rmsnorm_pass2_subgroup_size : transformer_spatial_rmsnorm_pass2_size;
    ComputeKernel kernel =
      ComputeKernel::build(device, spirvData, spirvWords, 2, sizeof(Pass2PC), specMap, specDat, cache);
    // logical: {partials, scalar} — FP32-only shader.
    kernel.bindingMap = {0, 1};
    kernel.localSizeX = (uint16_t)tileSize;
    kernel.debugName = useSubgroupVariant ? "SpatialRMSNorm.pass2.subgroup" : "SpatialRMSNorm.pass2";
    kernel.debugFp16 = false;
    return kernel;
  }

  ComputeKernel SpatialRMSNorm::buildPass3(VkDevice device, VkPipelineCache cache, bool fp16) {
    using namespace VulkanShaders;
    auto specMap = makeSpecMap(2);
    auto specDat = makeSpecData({(uint32_t)APPLY_ELTS_PER_THREAD, fp16 ? 1u : 0u});
    ComputeKernel kernel = ComputeKernel::build(
      device,
      transformer_spatial_rmsnorm_pass3,
      transformer_spatial_rmsnorm_pass3_size,
      fp16 ? 10 : 7,
      sizeof(Pass3PC),
      specMap,
      specDat,
      cache);
    // logical: {input, output, gamma, beta, mask, maskSum, scalar};
    // gamma/beta/maskSum/scalar stay FP32 in FP16 mode.
    kernel.bindingMap =
      fp16 ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 0, 1, 4} : std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6};
    kernel.debugName = "SpatialRMSNorm.pass3";
    kernel.debugFp16 = fp16;
    return kernel;
  }

  void SpatialRMSNorm::dispatchPass1(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* partials,
    const Pass1PC& pc,
    int maxBatchSize) {
    kernel.dispatch(
      ctx, {input, mask, partials}, &pc, sizeof(Pass1PC), (uint32_t)pc.numCHWWorkgroups, (uint32_t)maxBatchSize, 1u);
  }

  void SpatialRMSNorm::dispatchPass2(
    const CmdCtx& ctx,
    const ComputeKernel& kernel,
    VulkanBuffer* partials,
    VulkanBuffer* scalar,
    const Pass2PC& pc,
    int maxBatchSize) {
    kernel.dispatch(ctx, {partials, scalar}, &pc, sizeof(Pass2PC), 1u, (uint32_t)maxBatchSize, 1u);
  }

  void SpatialRMSNorm::dispatchPass3(
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
    int maxBatchSize) {
    const int chwSize = pc.numChannels * pc.paddedSpatialSize;
    const int numApplyGroups =
      (chwSize + PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD - 1) / (PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD);
    kernel.dispatch(
      ctx,
      {input, output, gamma, beta, mask, maskSum, scalar},
      &pc,
      sizeof(Pass3PC),
      (uint32_t)numApplyGroups,
      (uint32_t)maxBatchSize,
      1u);
  }

  std::array<ComputeKernel, 3>
  SpatialRMSNorm::build(VkDevice device, VkPipelineCache cache, bool fp16, int tileSize, bool useSubgroupVariant) {
    return {
      buildPass1(device, cache, fp16, tileSize),
      buildPass2(device, cache, tileSize, useSubgroupVariant),
      buildPass3(device, cache, fp16),
    };
  }

  void SpatialRMSNorm::dispatch(
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
    const int chwSize = pc.numChannels * pc.paddedSpatialSize;
    const int numCHWWorkgroups = (chwSize + tileSize - 1) / tileSize;
    const int tilesPerGroupPass2 = (numCHWWorkgroups + tileSize - 1) / tileSize;

    // Pass 1: partial sum-of-squares → partials
    Pass1PC pc1 = {pc.numChannels, pc.paddedSpatialSize, 1, numCHWWorkgroups};
    dispatchPass1(ctx, kernels[0], input, mask, partials, pc1, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, partials->buffer);

    // Pass 2: reduce partials → scalar (one per batch element)
    Pass2PC pc2 = {numCHWWorkgroups, tilesPerGroupPass2};
    dispatchPass2(ctx, kernels[1], partials, scalar, pc2, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, scalar->buffer);

    // Pass 3: normalize
    Pass3PC pc3 = {pc.numChannels, pc.paddedSpatialSize, pc.epsilon};
    dispatchPass3(ctx, kernels[2], input, output, gamma, beta, mask, maskSum, scalar, pc3, maxBatchSize);
  }

  ArrayView<TunableParam> SpatialRMSNorm::sharedParams() {
    static constexpr int32_t tileVals[] = {16, 32, 64, 128};
    static const std::array<TunableParam, 1> tab = {{
      {&VulkanTuneParams::spatialRMSNormTile, tileVals},
    }};
    return tab;
  }
  ArrayView<TunableParam> SpatialRMSNorm::params() const {
    return sharedParams();
  }

  SpatialRMSNorm::SpatialRMSNorm(int batchSize, int channels, int xySize)
    : problemBatchSize(batchSize), problemChannels(channels), problemXySize(xySize) {}

  double SpatialRMSNorm::estimatedFlopsPerDispatch(const VulkanTuneParams& cfg) const {
    if(problemBatchSize <= 0 || problemChannels <= 0 || problemXySize <= 0 || cfg.spatialRMSNormTile <= 0)
      return 0.0;
    int tile = cfg.spatialRMSNormTile;
    int chwSize = problemChannels * problemXySize;
    int numCHWWorkgroups = ceilDivInt(chwSize, tile);
    int numApplyGroups = ceilDivInt(chwSize, PASS3_LOCAL_X * APPLY_ELTS_PER_THREAD);

    double pass1 =
      3.0 * (double)chwSize + (double)numCHWWorkgroups * (double)(tile - 1);  // mask, square, sum, workgroup reduce
    double pass2 = (double)numCHWWorkgroups + (double)(tile - 1);
    double pass3Setup =
      (double)numApplyGroups * (double)PASS3_LOCAL_X * 4.0;  // repeated rms calculation per invocation
    double pass3Apply = 4.0 * (double)chwSize;
    return (double)problemBatchSize * (pass1 + pass2 + pass3Setup + pass3Apply);
  }

  bool SpatialRMSNorm::isConfigSupported(int32_t tileSize) {
    return tileSize > 0 && isPow2(tileSize);
  }

  bool SpatialRMSNorm::validate(const VulkanTuneParams& cfg, const VkPhysicalDeviceLimits& lim) const {
    if(!isConfigSupported(cfg.spatialRMSNormTile))
      return false;
    if(!workGroupFits((uint32_t)cfg.spatialRMSNormTile, 1u, 1u, lim))
      return false;
    // Shared memory: one array of spatialRMSNormTile floats (pass1/pass2 partials).
    if((size_t)cfg.spatialRMSNormTile * sizeof(float) > lim.maxComputeSharedMemorySize)
      return false;
    return true;
  }

  // bench() is defined in vulkantuner.cpp where TuningContext is a full type.

}  // namespace VulkanKernels

#endif  // USE_VULKAN_BACKEND

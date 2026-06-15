#ifndef NEURALNET_VULKAN_TUNER_H_
#define NEURALNET_VULKAN_TUNER_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "../core/commontypes.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"

struct VulkanDeviceInfo;

// VulkanTuneParams: all tuning knobs for the Vulkan backend in one place.
struct VulkanTuneParams {
  // ----- Tunable fields and tune-file metadata below. Tunable fields are all
  // int32_t so a single `int32_t VulkanTuneParams::*` pointer-to-member can
  // address any of them (see TunableParam in vulkankernels.h). -----

  // Winograd transform workgroup size (winograd_transform_nchw.glsl).
  int32_t winogradTransformLocalSizeX = 8;
  int32_t winogradTransformLocalSizeY = 8;
  // Metadata, not a tunable: identifies the selected Winograd GEMM layout that
  // the Winograd transform kernel workgroup sizes were last tuned against.
  std::string winogradTransformTunedLayout = "";
  // Metadata: canonical Vulkan GEMM variant environment filter signature used
  // when optional GEMM candidate validity and selection ranks were last updated.
  std::string gemmVariantFilterTunedSignature = "";

  // Winograd transform with fused BN+activation workgroup size
  // (winograd_bn_act_transform_nchw.glsl).
  int32_t winogradBNActTransformLocalSizeX = 8;
  int32_t winogradBNActTransformLocalSizeY = 8;

  // Winograd output untransform workgroup size (winograd_untransform_nchw.glsl).
  int32_t winogradUntransformLocalSizeX = 8;
  int32_t winogradUntransformLocalSizeY = 8;

  // 3x3 Winograd output tile size. 2 selects F(2,3), 4 selects F(4,3).
  int32_t winograd3x3OutTile = 2;

  // Per-shader workgroup sizes for the remaining 2-D dispatch kernels.
  // Each pair is independent (X covers the first dispatch dimension, Y the second)
  // following OpenCL's per-kernel tuning precedent.
  // gemm_strided_tiled.glsl: dispatches (ceil(M/(WG_X*4)), ceil(N/(WG_Y*RN)), batch).
  // Each thread owns a 4(m) x gemmStridedTiledRN(n) register block; gemmStridedTiledTileK is
  // the K-chunk staged into shared memory per pass.
  int32_t gemmStridedTiledLocalSizeX = 8;
  int32_t gemmStridedTiledLocalSizeY = 8;
  int32_t gemmStridedTiledTileK = 8;
  int32_t gemmStridedTiledRN = 4;
  int32_t gemmStridedTiledSelectionRank = 8;
  // gemm_direct_fp32.glsl: dispatches (batchElts, outC, batch)
  int32_t gemmDirectLocalSizeX = 8;
  int32_t gemmDirectLocalSizeY = 8;
  // Tiled GEMM (winograd_gemm_tiled.glsl) macro tile + register tile.
  // Workgroup is (winogradGemmM/4) x (winogradGemmN/winogradGemmRN); each thread owns a
  // 4(m) x winogradGemmRN(n) register block. validate() requires winogradGemmM % 4 == 0,
  // winogradGemmN % winogradGemmRN == 0, and winogradGemmRN % 4 == 0 (tileB is vec4-typed).
  int32_t winogradGemmM = 32;
  int32_t winogradGemmN = 32;
  int32_t winogradGemmK = 8;
  int32_t winogradGemmRN = 4;
  int32_t winogradGemmTiledSelectionRank = 8;

  // Deprecated full-FP16-compute tiled GEMM variants. Kept for tune-file
  // compatibility, but normalized off because FP16 accumulation is too
  // inaccurate for some models.
  int32_t enableWinogradGemmTiledFp16Compute = 0;
  int32_t enableGemmStridedTiledFp16Compute = 0;
  int32_t winogradGemmTiledFp16ComputeTunerValueValid = 0;
  int32_t gemmStridedTiledFp16ComputeTunerValueValid = 0;
  int32_t winogradGemmTiledFp16ComputeSelectionRank = 8;
  int32_t gemmStridedTiledFp16ComputeSelectionRank = 8;
  int32_t winogradGemmFp16ComputeM = 32;
  int32_t winogradGemmFp16ComputeN = 32;
  int32_t winogradGemmFp16ComputeK = 8;
  int32_t winogradGemmFp16ComputeRN = 4;
  int32_t gemmStridedFp16ComputeLocalSizeX = 8;
  int32_t gemmStridedFp16ComputeLocalSizeY = 8;
  int32_t gemmStridedFp16ComputeTileK = 8;
  int32_t gemmStridedFp16ComputeRN = 4;

  // DOT2 variants (winograd_gemm_dot2*.glsl / gemm_strided_dot2*.glsl).
  // Require the matching supportsDot2F16* flag and supportsFP16Storage. When
  // enabled, benchmarked against the current GEMM winner and used at runtime.
  int32_t enableWinogradGemmDot2 = 0;
  int32_t enableGemmStridedDot2 = 0;
  int32_t enableWinogradGemmDot2AccF16 = 0;
  int32_t enableGemmStridedDot2AccF16 = 0;
  int32_t winogradGemmDot2TunerValueValid = 0;
  int32_t gemmStridedDot2TunerValueValid = 0;
  int32_t winogradGemmDot2AccF16TunerValueValid = 0;
  int32_t gemmStridedDot2AccF16TunerValueValid = 0;
  int32_t winogradGemmDot2SelectionRank = 6;
  int32_t gemmStridedDot2SelectionRank = 6;
  int32_t winogradGemmDot2AccF16SelectionRank = 5;
  int32_t gemmStridedDot2AccF16SelectionRank = 5;
  // Winograd DOT2 tile config. Tunable spec constants; defaults match
  // VulkanKernels::DOT2_* (ggml F16 tile). BK is a fixed #define in the shaders
  // and is not tuned. Constraint (isConfigSupported):
  // dot2BlockSize == (dot2BM/dot2WM)*(dot2BN/dot2WN)*dot2Warp, dot2BM/dot2BN % 8 == 0,
  // dot2TM % 2 == 0.
  int32_t dot2BlockSize = 128;
  int32_t dot2BM = 64;
  int32_t dot2BN = 64;
  int32_t dot2WM = 32;
  int32_t dot2WN = 32;
  int32_t dot2WMIter = 2;
  int32_t dot2TM = 4;
  int32_t dot2TN = 2;
  int32_t dot2Warp = 32;
  // Strided DOT2 tile config. Kept separate from Winograd because 1x1 conv /
  // transformer matmul shapes can prefer a different tile on some GPUs.
  int32_t stridedDot2BlockSize = 128;
  int32_t stridedDot2BM = 64;
  int32_t stridedDot2BN = 64;
  int32_t stridedDot2WM = 32;
  int32_t stridedDot2WN = 32;
  int32_t stridedDot2WMIter = 2;
  int32_t stridedDot2TM = 4;
  int32_t stridedDot2TN = 2;
  int32_t stridedDot2Warp = 32;
  // DOT2 AccF16 variants use separate tile configs because FP16 accumulation
  // can prefer different occupancy/register tradeoffs than FP32 accumulation.
  int32_t dot2AccF16BlockSize = 128;
  int32_t dot2AccF16BM = 64;
  int32_t dot2AccF16BN = 64;
  int32_t dot2AccF16WM = 32;
  int32_t dot2AccF16WN = 32;
  int32_t dot2AccF16WMIter = 2;
  int32_t dot2AccF16TM = 4;
  int32_t dot2AccF16TN = 2;
  int32_t dot2AccF16Warp = 32;
  int32_t stridedDot2AccF16BlockSize = 128;
  int32_t stridedDot2AccF16BM = 64;
  int32_t stridedDot2AccF16BN = 64;
  int32_t stridedDot2AccF16WM = 32;
  int32_t stridedDot2AccF16WN = 32;
  int32_t stridedDot2AccF16WMIter = 2;
  int32_t stridedDot2AccF16TM = 4;
  int32_t stridedDot2AccF16TN = 2;
  int32_t stridedDot2AccF16Warp = 32;

  // Coopmat variant (winograd_gemm_coopmat.glsl / gemm_strided_coopmat.glsl).
  // Requires supportsCoopmat1F16 (KHR cooperative matrix + FP16 storage +
  // shaderFloat16 + subgroup-size guarantee). When enabled, benchmarked against
  // the tiled/DOT2 FP16 variants; winner is used at runtime.
  // TM/TN/TK are the device's reported coopmat fragment shape (MSize/NSize/KSize)
  // — set by the tuner from VulkanDeviceInfo::coopmatShapes, NOT freely swept.
  // The block/warp tiling (BlockSize/BM/BN/BK/WM/WN/Warp) is the tunable part.
  // Constraint (isConfigSupported): coopmatBlockSize ==
  // (coopmatBM/coopmatWM)*(coopmatBN/coopmatWN)*coopmatWarp; BM%WM==0, WM%TM==0,
  // BN%WN==0, WN%TN==0, BK%TK==0; BM/BN%8==0; Winograd TM%4==0 and BK/TN/TK%8==0;
  // strided BK%4==0 and TM/WM%8==0 for packed half8 A loads.
  int32_t enableWinogradGemmCoopmat = 0;
  int32_t enableGemmStridedCoopmat = 0;
  int32_t winogradGemmCoopmatTunerValueValid = 0;
  int32_t gemmStridedCoopmatTunerValueValid = 0;
  int32_t winogradGemmCoopmatSelectionRank = 4;
  int32_t gemmStridedCoopmatSelectionRank = 4;
  int32_t coopmatBlockSize = 128;
  int32_t coopmatBM = 64;
  int32_t coopmatBN = 64;
  int32_t coopmatBK = 16;
  int32_t coopmatWM = 32;
  int32_t coopmatWN = 32;
  int32_t coopmatTM = 16;
  int32_t coopmatTN = 16;
  int32_t coopmatTK = 16;
  int32_t coopmatWarp = 32;
  // Strided coopmat tile config. Kept separate from Winograd (see DOT2 rationale).
  int32_t stridedCoopmatBlockSize = 128;
  int32_t stridedCoopmatBM = 64;
  int32_t stridedCoopmatBN = 64;
  int32_t stridedCoopmatBK = 16;
  int32_t stridedCoopmatWM = 32;
  int32_t stridedCoopmatWN = 32;
  int32_t stridedCoopmatTM = 16;
  int32_t stridedCoopmatTN = 16;
  int32_t stridedCoopmatTK = 16;
  int32_t stridedCoopmatWarp = 32;

  // Coopmat AccF16 variant: same layouts as coopmat1, but C/Result fragments
  // and accumulation stay FP16. Separate tile params because FP16 accumulators
  // can have different supported shapes and preferred tiles.
  int32_t enableWinogradGemmCoopmatAccF16 = 0;
  int32_t enableGemmStridedCoopmatAccF16 = 0;
  int32_t winogradGemmCoopmatAccF16TunerValueValid = 0;
  int32_t gemmStridedCoopmatAccF16TunerValueValid = 0;
  int32_t winogradGemmCoopmatAccF16SelectionRank = 3;
  int32_t gemmStridedCoopmatAccF16SelectionRank = 3;
  int32_t coopmatAccF16BlockSize = 128;
  int32_t coopmatAccF16BM = 64;
  int32_t coopmatAccF16BN = 64;
  int32_t coopmatAccF16BK = 16;
  int32_t coopmatAccF16WM = 32;
  int32_t coopmatAccF16WN = 32;
  int32_t coopmatAccF16TM = 16;
  int32_t coopmatAccF16TN = 16;
  int32_t coopmatAccF16TK = 16;
  int32_t coopmatAccF16Warp = 32;
  int32_t stridedCoopmatAccF16BlockSize = 128;
  int32_t stridedCoopmatAccF16BM = 64;
  int32_t stridedCoopmatAccF16BN = 64;
  int32_t stridedCoopmatAccF16BK = 16;
  int32_t stridedCoopmatAccF16WM = 32;
  int32_t stridedCoopmatAccF16WN = 32;
  int32_t stridedCoopmatAccF16TM = 16;
  int32_t stridedCoopmatAccF16TN = 16;
  int32_t stridedCoopmatAccF16TK = 16;
  int32_t stridedCoopmatAccF16Warp = 32;

  // Coopmat2 variant (VK_NV_cooperative_matrix2 workgroup-scope cooperative
  // matrices). BM/BN/BK are flexible dimensions, constrained by the selected
  // device-reported granularities and workgroupInvocations. The Winograd path
  // further requires half8-aligned A/B tensor views.
  int32_t enableWinogradGemmCoopmat2 = 0;
  int32_t enableGemmStridedCoopmat2 = 0;
  int32_t winogradGemmCoopmat2TunerValueValid = 0;
  int32_t gemmStridedCoopmat2TunerValueValid = 0;
  int32_t winogradGemmCoopmat2SelectionRank = 2;
  int32_t gemmStridedCoopmat2SelectionRank = 2;
  int32_t coopmat2BlockSize = 128;
  int32_t coopmat2BM = 64;
  int32_t coopmat2BN = 64;
  int32_t coopmat2BK = 32;
  int32_t stridedCoopmat2BlockSize = 128;
  int32_t stridedCoopmat2BM = 64;
  int32_t stridedCoopmat2BN = 64;
  int32_t stridedCoopmat2BK = 32;

  // Coopmat2 AccF16 variant.
  int32_t enableWinogradGemmCoopmat2AccF16 = 0;
  int32_t enableGemmStridedCoopmat2AccF16 = 0;
  int32_t winogradGemmCoopmat2AccF16TunerValueValid = 0;
  int32_t gemmStridedCoopmat2AccF16TunerValueValid = 0;
  int32_t winogradGemmCoopmat2AccF16SelectionRank = 1;
  int32_t gemmStridedCoopmat2AccF16SelectionRank = 1;
  int32_t coopmat2AccF16BlockSize = 128;
  int32_t coopmat2AccF16BM = 64;
  int32_t coopmat2AccF16BN = 64;
  int32_t coopmat2AccF16BK = 32;
  int32_t stridedCoopmat2AccF16BlockSize = 128;
  int32_t stridedCoopmat2AccF16BM = 64;
  int32_t stridedCoopmat2AccF16BN = 64;
  int32_t stridedCoopmat2AccF16BK = 32;

  // Pooling / reduction workgroup X size (gpool_nchw_mask.glsl).
  // Runtime-tunable: the shader sizes its shared arrays from this spec constant
  // and local_size_x_id already drives local_size_x. Must be a power of 2 (the
  // butterfly reduction halves the span each step).
  int32_t gpoolXystride = 64;

  // Value-head pooling workgroup X size (value_head_pool_nchw.glsl). Same constraint.
  int32_t valueHeadPoolXystride = 64;

  // Spatial RMSNorm reduction tile (transformer_spatial_rmsnorm_pass{1,2}.glsl).
  // Same power-of-2 constraint (both passes share this tile size).
  int32_t spatialRMSNormTile = 64;

  // Tiled flash-attention block sizes (transformer_attention_tiled.glsl).
  // Runtime-tunable via spec constants (shared arrays sized from these).
  // Constraint: attnBlockKV <= attnBlockQ (host validation), matching the
  // shader's cooperative kMaskTile load shape.
  int32_t attnBlockQ = 32;
  int32_t attnBlockKV = 32;
  int32_t attnQPerThread = 1;

  bool operator==(const VulkanTuneParams& other) const;
  bool isValid() const;
  bool normalizeGemmVariantSelections();
  bool normalizeGemmVariantSelectionsForDevice(const VulkanDeviceInfo& deviceInfo, bool fp16Storage, bool fp16Compute);
  void inferLegacyGemmVariantTuningValidity();
  void inferLegacyGemmVariantSelectionRanks();

  static void save(const std::string& filename, const VulkanTuneParams& config);
  static VulkanTuneParams load(const std::string& filename);
};

inline int32_t winograd3x3OutTileFor(const VulkanTuneParams& config) {
  if(config.winograd3x3OutTile == 2 || config.winograd3x3OutTile == 4)
    return config.winograd3x3OutTile;
  throw StringError(
    "Invalid VulkanTuneParams::winograd3x3OutTile " + Global::intToString(config.winograd3x3OutTile) +
    " (expected 2 or 4)");
}

// VulkanTuner: per-GPU autotuning of the runtime-tunable VulkanTuneParams.
//
// This header holds only the Vulkan-API-free pieces (so command/tune.cpp can
// include it without Vulkan headers): the tune params, model info, device
// listing, file naming, and the command entry point. The actual tuning
// machinery (the TunableKernel interface in vulkankernels.h, the TuningContext
// device session in vulkanbackend.h, and the per-kernel benchmarks in
// vulkankernels.cpp) is free to depend on the Vulkan API.

namespace VulkanTuner {

  struct TuningContext;

  constexpr int DEFAULT_X_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_Y_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_BATCH_SIZE = 8;
  constexpr int DEFAULT_WINOGRAD_3X3_TILE_SIZE = 2;

  // ---- File path / naming conventions -------------------------------------

  std::string defaultDirectory(bool makeDir, const std::string& homeDataDirOverride);
  // tuneDeviceKey is a stable identifier for the *device category* used for
  // tuning: (vendorID, deviceID, driverVersion) formatted as a short hex string.
  // Same for two identical GPUs on the same host so they share the tuning
  // cache, while still separating genuinely different devices even when their
  // marketing deviceName collides.
  std::string defaultFileName(
    const std::string& gpuName,
    const std::string& tuneDeviceKey,
    int nnXLen,
    int nnYLen,
    const ModelDesc* modelDesc,
    bool fp16 = false);

  // Append a compact one-line summary of VulkanTuneParams to `out`.
  // Uses consistent x-separated tuples for all multi-value fields.
  void appendTuneParamsSummary(std::ostream& out, const VulkanTuneParams& config, const ModelDesc* modelDesc);

  std::string winogradTransformLayoutSignature(const VulkanTuneParams& config, bool fp16Storage);
  bool winogradTransformsTunedForCurrentLayout(const VulkanTuneParams& config, bool fp16Storage);
  void markWinogradTransformsTunedForCurrentLayout(VulkanTuneParams& config, bool fp16Storage);
  bool gemmVariantFilterTunedForCurrentHardware(const VulkanTuneParams& config, const VulkanDeviceInfo& deviceInfo);
  bool retuneWinogradTransformsForCurrentLayout(
    TuningContext& ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    VulkanTuneParams& tunedConfig);
  bool tuneGemmVariantsForCurrentHardware(
    TuningContext& ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    VulkanTuneParams& tunedConfig);

  // ---- Command-line entry point -------------------------------------------

  // Implements the `katago tuner` subcommand for the Vulkan backend (arg parsing,
  // device enumeration, per-GPU tune + save). Defined in vulkantuner.cpp so all
  // Vulkan tuning logic lives together; command/tune.cpp just forwards to it.
  int runTuneCommand(const std::vector<std::string>& args);
}  // namespace VulkanTuner

#endif  // NEURALNET_VULKAN_TUNER_H_

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

  // Identifies the selected Winograd GEMM layout that the NHWC transform and
  // untransform workgroup sizes were last tuned against. Their
  // workgroup sizes are independent, but their packed A/C geometry follows
  // the same selected Winograd GEMM layout.
  std::string nhwcWinogradTransformTunedLayout = "";
  // Metadata: canonical Vulkan acceleration selection-policy signature used
  // when optional candidate validity and winners were last updated. Runtime
  // environment filtering does not invalidate the stored measurements.
  std::string gemmVariantFilterTunedSignature = "";

  // Bitset of coarse kernel families whose base parameters have been tuned.
  // Optional GEMM variants retain their finer-grained *TunerValueValid fields.
  int32_t tunedKernelMask = 0;

  // Policy-independent weighted model times for Winograd GEMM, rounded to
  // microseconds. These reselect among already-tuned
  // Coopmat1/Coopmat2 or Dot2 kernels without benchmarking again. Zero means
  // not measured. FP16 accumulation must be at least 25% faster than the best
  // FP32-accumulation candidate in the selected accelerator tier.
  int32_t winogradGemmCoopmat1F32TimeUs = 0;
  int32_t winogradGemmCoopmat2F32TimeUs = 0;
  int32_t winogradGemmCoopmat1F16TimeUs = 0;
  int32_t winogradGemmCoopmat2F16TimeUs = 0;
  int32_t winogradGemmDot2F32TimeUs = 0;
  int32_t winogradGemmDot2F16TimeUs = 0;

  // Independent transpose geometry for the two directional tiled shaders.
  // Small tensors use the direct path through C=4; tiled tensors at or below
  // the direction-specific crossover use that direction's small tile.
  int32_t nchwToNhwcSmallTile = 16;
  int32_t nchwToNhwcLargeTile = 32;
  int32_t nchwToNhwcTileCrossover = 16;

  // NHWC Winograd transform/untransform workgroup sizes (winograd_transform_nhwc.glsl,
  // winograd_untransform_nhwc.glsl). One shared pair each covers both 3x3 and
  // 5x5 layers.
  int32_t nhwcWinogradTransformLocalSizeX = 8;
  int32_t nhwcWinogradTransformLocalSizeY = 8;
  int32_t nhwcWinogradUntransformLocalSizeX = 8;
  int32_t nhwcWinogradUntransformLocalSizeY = 8;
  // NHWC 3x3 Winograd output tile size. 2 selects F(2,3), 4 selects F(4,3).
  // (5x5 has no analogous choice; it is architecturally fixed to F(2,5).)
  int32_t nhwcWinograd3x3OutTile = 2;
  // Independent per-conv-size markers recording whether the NHWC Winograd path has
  // been confirmed usable/tuned for 3x3 and 5x5 layers respectively on this device.
  int32_t conv3x3NhwcWinogradTunerValueValid = 0;
  int32_t conv5x5NhwcWinogradTunerValueValid = 0;

  // Tiled NHWC strided-GEMM workgroup and register-tile geometry.
  int32_t gemmStridedTiledNhwcLocalSizeX = 8;
  int32_t gemmStridedTiledNhwcLocalSizeY = 8;
  int32_t gemmStridedTiledNhwcTileK = 8;
  int32_t gemmStridedTiledNhwcRN = 4;

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

  // DOT2 variants used by Winograd and native NHWC strided GEMM.
  // Require the matching supportsDot2F16* flag and supportsFP16Storage. When
  // enabled, benchmarked against the current GEMM winner and used at runtime.
  int32_t enableWinogradGemmDot2 = 0;
  int32_t enableWinogradGemmDot2AccF16 = 0;
  int32_t winogradGemmDot2TunerValueValid = 0;
  int32_t winogradGemmDot2AccF16TunerValueValid = 0;
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

  // Coopmat variant used by Winograd and native NHWC kernels.
  // Requires supportsCoopmat1F16 (KHR cooperative matrix + FP16 storage +
  // shaderFloat16 + subgroup-size guarantee). When enabled, benchmarked against
  // the tiled/DOT2 FP16 variants; winner is used at runtime.
  // TM/TN/TK are the device's reported coopmat fragment shape (MSize/NSize/KSize)
  // — set by the tuner from VulkanDeviceInfo::coopmatShapes, NOT freely swept.
  // The block/warp tiling (BlockSize/BM/BN/BK/WM/WN/Warp) is the tunable part.
  // Constraint (isConfigSupported): coopmat1BlockSize ==
  // (coopmat1BM/coopmat1WM)*(coopmat1BN/coopmat1WN)*coopmat1Warp; BM%WM==0, WM%TM==0,
  // BN%WN==0, WN%TN==0, BK%TK==0; BM/BN%8==0; Winograd TM%4==0 and BK/TN/TK%8==0;
  // strided BK%4==0 and TM/WM%8==0 for packed half8 A loads.
  int32_t enableWinogradGemmCoopmat1 = 0;
  int32_t winogradGemmCoopmat1TunerValueValid = 0;
  int32_t coopmat1BlockSize = 128;
  int32_t coopmat1BM = 64;
  int32_t coopmat1BN = 64;
  int32_t coopmat1BK = 16;
  int32_t coopmat1WM = 32;
  int32_t coopmat1WN = 32;
  int32_t coopmat1TM = 16;
  int32_t coopmat1TN = 16;
  int32_t coopmat1TK = 16;
  int32_t coopmat1Warp = 32;

  // Native-NHWC 3x3 implicit-GEMM tile. This kernel's gather-heavy A path has
  // very different occupancy requirements from the generic strided GEMM, so
  // sharing its tile can severely regress small-batch latency.
  int32_t conv3x3NhwcCoopmat1TunerValueValid = 0;
  int32_t conv3x3NhwcCoopmat1BlockSize = 128;
  int32_t conv3x3NhwcCoopmat1BM = 64;
  int32_t conv3x3NhwcCoopmat1BN = 64;
  int32_t conv3x3NhwcCoopmat1BK = 32;
  int32_t conv3x3NhwcCoopmat1WM = 64;
  int32_t conv3x3NhwcCoopmat1WN = 16;
  int32_t conv3x3NhwcCoopmat1TM = 16;
  int32_t conv3x3NhwcCoopmat1TN = 8;
  int32_t conv3x3NhwcCoopmat1TK = 8;
  int32_t conv3x3NhwcCoopmat1Warp = 32;

  // 5x5 uses the same implicit-GEMM shader source but has a much larger K and
  // therefore needs an independently tuned NHWC tile.
  int32_t conv5x5NhwcCoopmat1TunerValueValid = 0;
  int32_t conv5x5NhwcCoopmat1BlockSize = 128;
  int32_t conv5x5NhwcCoopmat1BM = 64;
  int32_t conv5x5NhwcCoopmat1BN = 64;
  int32_t conv5x5NhwcCoopmat1BK = 32;
  int32_t conv5x5NhwcCoopmat1WM = 64;
  int32_t conv5x5NhwcCoopmat1WN = 16;
  int32_t conv5x5NhwcCoopmat1TM = 16;
  int32_t conv5x5NhwcCoopmat1TN = 8;
  int32_t conv5x5NhwcCoopmat1TK = 8;
  int32_t conv5x5NhwcCoopmat1Warp = 32;

  // FP16-accumulation native convolution has a distinct accumulator fragment
  // and gather/occupancy tradeoff, so it must not inherit the 1x1 GEMM tile.
  int32_t conv3x3NhwcCoopmat1AccF16TunerValueValid = 0;
  int32_t conv3x3NhwcCoopmat1AccF16BlockSize = 128;
  int32_t conv3x3NhwcCoopmat1AccF16BM = 64;
  int32_t conv3x3NhwcCoopmat1AccF16BN = 64;
  int32_t conv3x3NhwcCoopmat1AccF16BK = 16;
  int32_t conv3x3NhwcCoopmat1AccF16WM = 32;
  int32_t conv3x3NhwcCoopmat1AccF16WN = 32;
  int32_t conv3x3NhwcCoopmat1AccF16TM = 16;
  int32_t conv3x3NhwcCoopmat1AccF16TN = 16;
  int32_t conv3x3NhwcCoopmat1AccF16TK = 16;
  int32_t conv3x3NhwcCoopmat1AccF16Warp = 32;

  int32_t conv5x5NhwcCoopmat1AccF16TunerValueValid = 0;
  int32_t conv5x5NhwcCoopmat1AccF16BlockSize = 128;
  int32_t conv5x5NhwcCoopmat1AccF16BM = 64;
  int32_t conv5x5NhwcCoopmat1AccF16BN = 64;
  int32_t conv5x5NhwcCoopmat1AccF16BK = 16;
  int32_t conv5x5NhwcCoopmat1AccF16WM = 32;
  int32_t conv5x5NhwcCoopmat1AccF16WN = 32;
  int32_t conv5x5NhwcCoopmat1AccF16TM = 16;
  int32_t conv5x5NhwcCoopmat1AccF16TN = 16;
  int32_t conv5x5NhwcCoopmat1AccF16TK = 16;
  int32_t conv5x5NhwcCoopmat1AccF16Warp = 32;

  // Coopmat AccF16 variant: same layouts as coopmat1, but C/Result fragments
  // and accumulation stay FP16. Separate tile params because FP16 accumulators
  // can have different supported shapes and preferred tiles.
  int32_t enableWinogradGemmCoopmat1AccF16 = 0;
  int32_t winogradGemmCoopmat1AccF16TunerValueValid = 0;
  int32_t coopmat1AccF16BlockSize = 128;
  int32_t coopmat1AccF16BM = 64;
  int32_t coopmat1AccF16BN = 64;
  int32_t coopmat1AccF16BK = 16;
  int32_t coopmat1AccF16WM = 32;
  int32_t coopmat1AccF16WN = 32;
  int32_t coopmat1AccF16TM = 16;
  int32_t coopmat1AccF16TN = 16;
  int32_t coopmat1AccF16TK = 16;
  int32_t coopmat1AccF16Warp = 32;

  // Native-NHWC 1x1 GEMM tile and validity state.
  int32_t gemmStridedNhwcCoopmat1TunerValueValid = 0;
  int32_t nhwcStridedCoopmat1BlockSize = 128;
  int32_t nhwcStridedCoopmat1BM = 64;
  int32_t nhwcStridedCoopmat1BN = 64;
  int32_t nhwcStridedCoopmat1BK = 16;
  int32_t nhwcStridedCoopmat1WM = 32;
  int32_t nhwcStridedCoopmat1WN = 32;
  int32_t nhwcStridedCoopmat1TM = 16;
  int32_t nhwcStridedCoopmat1TN = 16;
  int32_t nhwcStridedCoopmat1TK = 16;
  int32_t nhwcStridedCoopmat1Warp = 32;
  int32_t gemmStridedNhwcCoopmat1AccF16TunerValueValid = 0;
  int32_t nhwcStridedCoopmat1AccF16BlockSize = 128;
  int32_t nhwcStridedCoopmat1AccF16BM = 64;
  int32_t nhwcStridedCoopmat1AccF16BN = 64;
  int32_t nhwcStridedCoopmat1AccF16BK = 16;
  int32_t nhwcStridedCoopmat1AccF16WM = 32;
  int32_t nhwcStridedCoopmat1AccF16WN = 32;
  int32_t nhwcStridedCoopmat1AccF16TM = 16;
  int32_t nhwcStridedCoopmat1AccF16TN = 16;
  int32_t nhwcStridedCoopmat1AccF16TK = 16;
  int32_t nhwcStridedCoopmat1AccF16Warp = 32;
  // Native-NHWC coopmat2 candidates are tuned independently, then compete
  // with the coopmat1 candidates through the common aggregate selector.
  int32_t gemmStridedNhwcCoopmat2TunerValueValid = 0;
  int32_t nhwcStridedCoopmat2BlockSize = 128;
  int32_t nhwcStridedCoopmat2BM = 64;
  int32_t nhwcStridedCoopmat2BN = 64;
  int32_t nhwcStridedCoopmat2BK = 32;
  int32_t gemmStridedNhwcCoopmat2AccF16TunerValueValid = 0;
  int32_t nhwcStridedCoopmat2AccF16BlockSize = 128;
  int32_t nhwcStridedCoopmat2AccF16BM = 64;
  int32_t nhwcStridedCoopmat2AccF16BN = 64;
  int32_t nhwcStridedCoopmat2AccF16BK = 32;
  // 0=coopmat1, 1=coopmat2 within each accumulation type. The final mode
  // selector chooses FP32 (0) or FP16 (1) accumulation between those winners.
  int32_t nhwcGemmF32UseCoopmat2 = 0;
  int32_t nhwcGemmUseCoopmat2 = 0;
  int32_t nhwcGemmUseCoopmatAccF16 = 0;
  // Policy-independent weighted model times, rounded to microseconds. These
  // let a GEMM_VARIANTS filter change reselect among already-tuned kernels
  // without dispatching another benchmark. Zero means not measured.
  int32_t nhwcGemmCoopmat1F32TimeUs = 0;
  int32_t nhwcGemmCoopmat2F32TimeUs = 0;
  int32_t nhwcGemmCoopmat1F16TimeUs = 0;
  int32_t nhwcGemmCoopmat2F16TimeUs = 0;

  // Native-NHWC 1x1 GEMM DOT2 variants. There is no bk/tk field because
  // DOT2's BK is a fixed shader define.
  int32_t gemmStridedNhwcDot2TunerValueValid = 0;
  int32_t nhwcStridedDot2BlockSize = 128;
  int32_t nhwcStridedDot2BM = 64;
  int32_t nhwcStridedDot2BN = 64;
  int32_t nhwcStridedDot2WM = 32;
  int32_t nhwcStridedDot2WN = 32;
  int32_t nhwcStridedDot2WMIter = 2;
  int32_t nhwcStridedDot2TM = 4;
  int32_t nhwcStridedDot2TN = 2;
  int32_t nhwcStridedDot2Warp = 32;
  int32_t nhwcGemmUseDot2 = 0;

  int32_t gemmStridedNhwcDot2AccF16TunerValueValid = 0;
  int32_t nhwcStridedDot2AccF16BlockSize = 128;
  int32_t nhwcStridedDot2AccF16BM = 64;
  int32_t nhwcStridedDot2AccF16BN = 64;
  int32_t nhwcStridedDot2AccF16WM = 32;
  int32_t nhwcStridedDot2AccF16WN = 32;
  int32_t nhwcStridedDot2AccF16WMIter = 2;
  int32_t nhwcStridedDot2AccF16TM = 4;
  int32_t nhwcStridedDot2AccF16TN = 2;
  int32_t nhwcStridedDot2AccF16Warp = 32;
  int32_t nhwcGemmUseDot2AccF16 = 0;
  int32_t nhwcGemmDot2F32TimeUs = 0;
  int32_t nhwcGemmDot2F16TimeUs = 0;

  int32_t conv3x3NhwcCoopmat2TunerValueValid = 0;

  int32_t conv3x3NhwcCoopmat2BlockSize = 128;
  int32_t conv3x3NhwcCoopmat2BM = 64;
  int32_t conv3x3NhwcCoopmat2BN = 64;
  int32_t conv3x3NhwcCoopmat2BK = 32;
  int32_t nhwcConv3x3UseCoopmat2 = 0;

  int32_t conv5x5NhwcCoopmat2TunerValueValid = 0;
  int32_t conv5x5NhwcCoopmat2BlockSize = 128;
  int32_t conv5x5NhwcCoopmat2BM = 64;
  int32_t conv5x5NhwcCoopmat2BN = 64;
  int32_t conv5x5NhwcCoopmat2BK = 32;
  int32_t nhwcConv5x5UseCoopmat2 = 0;

  int32_t conv3x3NhwcCoopmat2AccF16TunerValueValid = 0;
  int32_t conv3x3NhwcCoopmat2AccF16BlockSize = 128;
  int32_t conv3x3NhwcCoopmat2AccF16BM = 64;
  int32_t conv3x3NhwcCoopmat2AccF16BN = 64;
  int32_t conv3x3NhwcCoopmat2AccF16BK = 32;

  int32_t conv5x5NhwcCoopmat2AccF16TunerValueValid = 0;
  int32_t conv5x5NhwcCoopmat2AccF16BlockSize = 128;
  int32_t conv5x5NhwcCoopmat2AccF16BM = 64;
  int32_t conv5x5NhwcCoopmat2AccF16BN = 64;
  int32_t conv5x5NhwcCoopmat2AccF16BK = 32;

  // Coopmat2 variant (VK_NV_cooperative_matrix2 workgroup-scope cooperative
  // matrices). BM/BN/BK are flexible dimensions, constrained by the selected
  // device-reported granularities and workgroupInvocations. The Winograd path
  // further requires half8-aligned A/B tensor views.
  int32_t enableWinogradGemmCoopmat2 = 0;
  int32_t winogradGemmCoopmat2TunerValueValid = 0;
  int32_t coopmat2BlockSize = 128;
  int32_t coopmat2BM = 64;
  int32_t coopmat2BN = 64;
  int32_t coopmat2BK = 32;

  // Coopmat2 AccF16 variant.
  int32_t enableWinogradGemmCoopmat2AccF16 = 0;
  int32_t winogradGemmCoopmat2AccF16TunerValueValid = 0;
  int32_t coopmat2AccF16BlockSize = 128;
  int32_t coopmat2AccF16BM = 64;
  int32_t coopmat2AccF16BN = 64;
  int32_t coopmat2AccF16BK = 32;

  // Pooling / reduction workgroup X size.
  // Runtime-tunable: the shader sizes its shared arrays from this spec constant
  // and local_size_x_id already drives local_size_x. Must be a power of 2 (the
  // butterfly reduction halves the span each step).
  int32_t gpoolNhwcXystride = 64;

  // Value-head pooling workgroup X size. Same constraint.
  int32_t valueHeadPoolNhwcXystride = 64;

  // Spatial RMSNorm reduction tile (transformer_spatial_rmsnorm_pass{1,2}.glsl).
  // Same power-of-2 constraint (both passes share this tile size).
  int32_t spatialRMSNormNhwcTile = 64;

  // SwiGLU pointwise activation workgroup size (transformer_swiglu.glsl).
  // The tuner selects among 64, 128, 256, and 512 threads.
  int32_t swiGLUTunerValueValid = 0;
  int32_t swiGLULocalSizeX = 64;

  // Tiled flash-attention block sizes (transformer_attention_tiled.glsl).
  // Runtime-tunable via spec constants (shared arrays sized from these).
  // Constraint: attnBlockKV <= attnBlockQ (host validation), matching the
  // shader's cooperative kMaskTile load shape.
  int32_t attnBlockQ = 32;
  int32_t attnBlockKV = 32;
  int32_t attnQPerThread = 1;
  int32_t attnNhwcBlockQ = 32;
  int32_t attnNhwcBlockKV = 32;
  int32_t attnNhwcQPerThread = 1;

  // Portable NHWC VK_KHR_cooperative_matrix attention.
  int32_t attnNhwcVariantPolicyVersion = 4;
  int32_t attnNhwcCoopmat1TunerValueValid = 0;
  int32_t attnNhwcUseCoopmat1 = 0;
  int32_t attnNhwcTiledTimeUs = 0;
  int32_t attnNhwcCoopmat1TimeUs = 0;
  int32_t attnNhwcCoopmat1BlockSize = 128;
  // blockQ is the workgroup width for this kernel and must match blockSize.
  // Keep the default self-consistent even when this optional path is not tuned.
  int32_t attnNhwcCoopmat1BlockQ = 128;
  int32_t attnNhwcCoopmat1BlockKV = 32;
  int32_t attnNhwcCoopmat1TM = 16;
  int32_t attnNhwcCoopmat1TN = 16;
  int32_t attnNhwcCoopmat1TK = 16;
  int32_t attnNhwcCoopmat1Warp = 32;
  // Enable ggml-style direct global K/V coopmat loads when the specialized
  // sequence length has no partial KV tile. The host falls back safely when it does.
  int32_t attnNhwcCoopmat1DirectKV = 0;
  // Split-K: number of workgroup-z slices the KV sequence is divided into.
  // 1 = no split (original behavior). >1 = each chunk writes partial O + stats,
  // a resolve shader merges them. Helps at low batch where the grid is small.
  int32_t attnNhwcCoopmat1KVChunkCount = 1;
  // Independently tuned batch-1 split-K attention configuration. The regular
  // coopmat1 fields above remain the unsplit path for larger batches.
  int32_t attnNhwcCoopmat1SplitKTunerValueValid = 0;
  int32_t attnNhwcCoopmat1SplitKBlockSize = 128;
  int32_t attnNhwcCoopmat1SplitKBlockQ = 128;
  int32_t attnNhwcCoopmat1SplitKBlockKV = 32;
  int32_t attnNhwcCoopmat1SplitKTM = 16;
  int32_t attnNhwcCoopmat1SplitKTN = 16;
  int32_t attnNhwcCoopmat1SplitKTK = 16;
  int32_t attnNhwcCoopmat1SplitKWarp = 32;
  int32_t attnNhwcCoopmat1SplitKDirectKV = 0;
  int32_t attnNhwcCoopmat1SplitKKVChunkCount = 1;
  // Split-K is selected for every batch <= this cutoff. Zero disables it.
  int32_t attnNhwcCoopmat1SplitKCutoffBatch = 0;

  // VK_EXT_cooperative_matrix_maintenance1 is a separately measured third
  // cooperative attention family. Keep its regular and split-K choices apart
  // from the portable KHR path so old tune files remain useful as fallbacks.
  int32_t attnNhwcCoopmatMaintenance1TunerValueValid = 0;
  int32_t attnNhwcUseCoopmatMaintenance1 = 0;
  int32_t attnNhwcCoopmatMaintenance1TimeUs = 0;
  int32_t attnNhwcCoopmatMaintenance1BlockSize = 128;
  int32_t attnNhwcCoopmatMaintenance1BlockQ = 128;
  int32_t attnNhwcCoopmatMaintenance1BlockKV = 32;
  int32_t attnNhwcCoopmatMaintenance1TM = 16;
  int32_t attnNhwcCoopmatMaintenance1TN = 16;
  int32_t attnNhwcCoopmatMaintenance1TK = 16;
  int32_t attnNhwcCoopmatMaintenance1Warp = 32;
  int32_t attnNhwcCoopmatMaintenance1DirectKV = 0;
  int32_t attnNhwcCoopmatMaintenance1SplitKTunerValueValid = 0;
  int32_t attnNhwcCoopmatMaintenance1SplitKBlockSize = 128;
  int32_t attnNhwcCoopmatMaintenance1SplitKBlockQ = 128;
  int32_t attnNhwcCoopmatMaintenance1SplitKBlockKV = 32;
  int32_t attnNhwcCoopmatMaintenance1SplitKTM = 16;
  int32_t attnNhwcCoopmatMaintenance1SplitKTN = 16;
  int32_t attnNhwcCoopmatMaintenance1SplitKTK = 16;
  int32_t attnNhwcCoopmatMaintenance1SplitKWarp = 32;
  int32_t attnNhwcCoopmatMaintenance1SplitKDirectKV = 0;
  int32_t attnNhwcCoopmatMaintenance1SplitKKVChunkCount = 1;
  int32_t attnNhwcCoopmatMaintenance1SplitKCutoffBatch = 0;

  // Workgroup-scope VK_NV_cooperative_matrix2 attention, FP16 inputs with
  // FP32 accumulation. blockQ is intentionally equal to blockSize: one lane
  // owns one softmax row and the matrix spans the whole workgroup.
  int32_t attnNhwcCoopmat2TunerValueValid = 0;
  int32_t attnNhwcUseCoopmat2 = 0;
  int32_t attnNhwcCoopmat2TimeUs = 0;
  int32_t attnNhwcCoopmat2BlockSize = 128;
  int32_t attnNhwcCoopmat2BlockQ = 128;
  int32_t attnNhwcCoopmat2BlockKV = 32;

  // NHWC SPV_VALVE_mixed_float_dot_product attention, FP16 inputs with FP32
  // accumulation. This tier is tuned only when coopmat1 AccF32 is unavailable
  // after applying KATAGO_VULKAN_ACCEL_VARIANTS.
  int32_t attnNhwcDot2TunerValueValid = 0;
  int32_t attnNhwcUseDot2 = 0;
  int32_t attnNhwcDot2TimeUs = 0;
  int32_t attnNhwcDot2BlockSize = 128;
  int32_t attnNhwcDot2BlockQ = 128;
  int32_t attnNhwcDot2BlockKV = 32;

  bool operator==(const VulkanTuneParams& other) const;
  bool isValid() const;
  bool normalizeGemmVariantSelectionsForDevice(const VulkanDeviceInfo& deviceInfo, bool fp16Storage, bool fp16Compute);

  static void save(const std::string& filename, const VulkanTuneParams& config);
  static VulkanTuneParams load(const std::string& filename);
};

inline int32_t nhwcWinograd3x3OutTileFor(const VulkanTuneParams& config) {
  if(config.nhwcWinograd3x3OutTile == 2 || config.nhwcWinograd3x3OutTile == 4)
    return config.nhwcWinograd3x3OutTile;
  throw StringError(
    "Invalid VulkanTuneParams::nhwcWinograd3x3OutTile " + Global::intToString(config.nhwcWinograd3x3OutTile) +
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

  enum TunedKernelFamily : int32_t {
    TUNED_GEMM_DIRECT = 1 << 0,
    TUNED_TRANSFORMER = 1 << 4,
    TUNED_NHWC_REDUCTIONS = 1 << 5,
    TUNED_NHWC_GEMM = 1 << 6,
    TUNED_NHWC_CONV3X3 = 1 << 7,
    // Versioned marker for NHWC GEMM selection using shared weights and
    // the model's overwrite/residual-add dispatch mix. Its absence in an old
    // tune file triggers only the incremental GEMM variant update.
    TUNED_NHWC_GEMM_RUNTIME_MODEL = 1 << 9,
    TUNED_NHWC_CONV5X5 = 1 << 10,
    // Tiled NHWC strided GEMM has its own global A-load and C-store tuning.
    TUNED_NHWC_STRIDED = 1 << 13,
    // NHWC Winograd fallback convolution (winogradTransformNhwc/winogradUntransformNhwc).
    // Only relevant on non-coopmat NHWC devices, where it is the non-coopmat fallback
    // convolution in place of the tiled implicit-GEMM shader.
    TUNED_NHWC_WINOGRAD = 1 << 14,
    TUNED_NCHW_TO_NHWC = 1 << 15,
    TUNED_NHWC_TRANSFORMER = 1 << 17,
    TUNED_NHWC_ATTENTION_VARIANT = 1 << 18,
    TUNED_SWIGLU = 1 << 19,
  };

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
  bool nhwcWinogradTransformsTunedForCurrentLayout(const VulkanTuneParams& config, bool fp16Storage);
  void markNhwcWinogradTransformsTunedForCurrentLayout(VulkanTuneParams& config, bool fp16Storage);
  bool gemmVariantFilterTunedForCurrentHardware(const VulkanTuneParams& config, const VulkanDeviceInfo& deviceInfo);
  bool gemmVariantTuningCompleteForCurrentHardware(
    const VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute,
    int32_t requiredMask,
    bool useNhwc);
  int32_t requiredKernelMask(
    const ModelDesc* modelDesc,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute,
    bool useNhwc,
    bool full);
  bool retuneNhwcWinogradTransformsForCurrentLayout(
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

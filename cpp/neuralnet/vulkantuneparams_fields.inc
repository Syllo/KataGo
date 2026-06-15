// Single source of truth for every VulkanTuneParams field. This file is meant to
// be edited by hand: adding a field here automatically flows into the member
// declarations, operator==, save/load, the reconcile resets, and the TunableParam
// candidate tables.
//
// Each field is one line:
//   VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES)
// where CANDIDATES is the parenthesized tuner candidate list, or () when the field
// has no candidates.
//
// Selection: the includer #define's VULKAN_TUNE_PARAM_FIELD as the per-field body
// (default &VulkanTuneParams::NAME, for the reconcile resets) plus one or more of
// the trigger macros below; the per-group #if guards pick out the matching fields.
// The triggers and VULKAN_TUNE_PARAM_FIELD are #undef'd at the bottom of this
// file, so each include site is just a `#define VULKAN_TUNE_PARAM_<TRIGGER>` line
// followed by the #include with nothing after it.
//
// Triggers:
//   VULKAN_TUNE_PARAM_ALL                 every field.
//   VULKAN_TUNE_PARAM_ALL_ACCELERATED      every fp16 accelerator field.
//   VULKAN_TUNE_PARAM_KERNEL_<K>           one kernel's fields.
//   VULKAN_TUNE_PARAM_KERNEL_<K>_TIMEUS    that kernel's TimeUs fields only.
//   VULKAN_TUNE_PARAM_KERNEL_<K>_FULL      that kernel's fields and TimeUs fields.
//
#ifndef VULKAN_TUNE_PARAM_FIELD
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) &VulkanTuneParams::NAME,
#endif

// ---------------------------------------------------------------------------
// Tile-group shorthands.
//
// 20 of the field groups below are accelerator tile geometries with identical
// member sequences, differing only in their name prefix and per-group defaults:
//
//   coopmat1: WorkgroupSize BM BN BK SGM SGN TM TN TK SubgroupSize   (10)
//   coopmat2: WorkgroupSize BM BN BK                                 (4)
//   dot2:     WorkgroupSize BM BN SGM SGN SGMIter TM TN SubgroupSize (9)
//
// These macros expand one group from its prefix plus defaults, so a group is a
// single line instead of 4/9/10 near-identical ones. The member sequence here is
// the same one the k*TileFields arrays in vulkantuner_shared.h list in tile-member
// order, and the static_asserts there pin the counts -- so if a sequence below
// ever changes, those assertions fail rather than the mismatch going unnoticed.
//
// Tile fields never carry tuner candidate lists (their candidates come from the
// curated per-family tile tables in vulkantuner.cpp, not from the .inc), hence
// the unconditional () candidate argument.
// ---------------------------------------------------------------------------
#define VULKAN_TUNE_COOPMAT1_TILE_GROUP(PREFIX, D_WG, D_BM, D_BN, D_BK, D_SGM, D_SGN, D_TM, D_TN, D_TK, D_SG) \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##WorkgroupSize, D_WG, ())                                           \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BM, D_BM, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BN, D_BN, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BK, D_BK, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SGM, D_SGM, ())                                                    \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SGN, D_SGN, ())                                                    \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##TM, D_TM, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##TN, D_TN, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##TK, D_TK, ())                                                      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SubgroupSize, D_SG, ())

#define VULKAN_TUNE_COOPMAT2_TILE_GROUP(PREFIX, D_WG, D_BM, D_BN, D_BK)  \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##WorkgroupSize, D_WG, ())      \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BM, D_BM, ())                 \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BN, D_BN, ())                 \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BK, D_BK, ())

#define VULKAN_TUNE_DOT2_TILE_GROUP(PREFIX, D_WG, D_BM, D_BN, D_SGM, D_SGN, D_SGMI, D_TM, D_TN, D_SG) \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##WorkgroupSize, D_WG, ())                                   \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BM, D_BM, ())                                              \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##BN, D_BN, ())                                              \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SGM, D_SGM, ())                                            \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SGN, D_SGN, ())                                            \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SGMIter, D_SGMI, ())                                       \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##TM, D_TM, ())                                              \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##TN, D_TN, ())                                              \
  VULKAN_TUNE_PARAM_FIELD(int32_t, PREFIX##SubgroupSize, D_SG, ())

// ---------------------------------------------------------------------------
// Fields, in exact current VulkanTuneParams declaration order.
// ---------------------------------------------------------------------------

#if defined(VULKAN_TUNE_PARAM_ALL)
// Identifies the selected Winograd GEMM layout that the NHWC transform and
// untransform workgroup sizes were last tuned against. Their workgroup sizes
// are independent, but both act on the GEMM's A/C operands: the transform
// writes the packed A input of the Winograd GEMM, and the untransform must
// match the layout of the GEMM's C output, so the signature encodes that A/C
// geometry. Comparing it against the current layout tells us whether to
// invalidate the tuned transform/untransform workgroup sizes.
VULKAN_TUNE_PARAM_FIELD(std::string, nhwcWinogradTransformTunedLayout, "", ())
// Validity bitset: one bit per tunable kernel/variant, recording whether
// that kernel's tuning has been measured for this device. Family sets are
// composed from these bits via the VulkanTuner::*FlavorBits() helpers.
VULKAN_TUNE_PARAM_FIELD(int64_t, tunedKernelMask, 0, ())
// Subset of tunedKernelMask recording kernels whose most recent tuning attempt
// found no valid candidate. The only legal (tunedKernelMask, invalidKernelMask)
// pairs are (0,0), (1,0) and (1,1); a cleared tuned bit clears its invalid bit.
VULKAN_TUNE_PARAM_FIELD(int64_t, invalidKernelMask, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_FULL)
// Policy-independent weighted model times for the Winograd GEMM variants
// (winograd_gemm_coopmat{1,2}_accf{32,16}.glsl, winograd_gemm_dot2_accf{32,16}.glsl).
// Zero means not measured.
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmCoopmat1F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_TIMEUS)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmCoopmat2F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_TIMEUS)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmCoopmat1F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmCoopmat2F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmDot2F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_TIMEUS)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmDot2F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_NCHW_TO_NHWC)
// NCHW->NHWC transpose (nchw_to_nhwc_smallchannel.glsl, nchw_to_nhwc_tiled.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, nchwToNhwcSmallTile, 16, (8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, nchwToNhwcLargeTile, 32, (16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, nchwToNhwcTileCrossover, 16, (8, 16, 32))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_TRANSFORM_NHWC)
// NHWC Winograd transform/untransform workgroup sizes (winograd_transform_nhwc.glsl,
// winograd_untransform_nhwc.glsl). Covers both 3x3 and 5x5 layers.
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcWinogradTransformLocalSizeX, 8, (4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcWinogradTransformLocalSizeY, 8, (4, 8, 16, 32))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_UNTRANSFORM_NHWC)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcWinogradUntransformLocalSizeX, 8, (4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcWinogradUntransformLocalSizeY, 8, (4, 8, 16, 32))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_3X3_OUT_TILE)
// NHWC 3x3 Winograd output tile size (winograd_transform_nhwc.glsl).
// 2 selects F(2,3), 4 selects F(4,3). (5x5 has no analogous choice; it is
// architecturally fixed to F(2,5).)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcWinograd3x3OutTile, 2, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_TILED)
// Tiled NHWC strided-GEMM workgroup and register-tile geometry
// (gemm_strided_tiled_nhwc.glsl).
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmStridedTiledNhwcLocalSizeX, 8, (4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmStridedTiledNhwcLocalSizeY, 8, (4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmStridedTiledNhwcTileK, 8, (4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmStridedTiledNhwcRN, 4, (4, 8))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_DIRECT)
// Direct GEMM (gemm_direct_fp32.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmDirectLocalSizeX, 8, (1, 2, 4, 8, 16, 32))
VULKAN_TUNE_PARAM_FIELD(int32_t, gemmDirectLocalSizeY, 8, (1, 2, 4, 8, 16, 32))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_TILED)
// Tiled GEMM (winograd_gemm_tiled.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmM, 32, (32, 64, 128, 256))
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmN, 32, (32, 64, 128, 256))
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmK, 8, (8, 16, 32, 64))
VULKAN_TUNE_PARAM_FIELD(int32_t, winogradGemmRN, 4, (4, 8))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2)
// Winograd DOT2 tile config (winograd_gemm_dot2_accf32.glsl)
VULKAN_TUNE_DOT2_TILE_GROUP(dot2, 128, 64, 64, 32, 32, 2, 4, 2, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16)
// DOT2 AccF16 variants (winograd_gemm_dot2_accf16.glsl)
VULKAN_TUNE_DOT2_TILE_GROUP(dot2AccF16, 128, 64, 64, 32, 32, 2, 4, 2, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_FULL)
// Coopmat variant used by Winograd and native NHWC kernels
// (winograd_gemm_coopmat1_accf32.glsl).
VULKAN_TUNE_COOPMAT1_TILE_GROUP(coopmat1, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_FULL)
// NHWC 3x3 implicit GEMM (conv_implicit_gemm_coopmat1_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(conv3x3NhwcCoopmat1, 128, 64, 64, 32, 64, 16, 16, 8, 8, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1)
// NHWC 5x5 implicit GEMM (conv_implicit_gemm_coopmat1_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(conv5x5NhwcCoopmat1, 128, 64, 64, 32, 64, 16, 16, 8, 8, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_FULL)
// NHWC 3x3 implicit GEMM, FP16 accumulation (conv_implicit_gemm_coopmat1_accf32_nhwc.glsl
// compiled with KATAGO_VULKAN_ACCUMULATE_FP16=1)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(conv3x3NhwcCoopmat1AccF16, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_FULL)
// NHWC 5x5 implicit GEMM, FP16 accumulation (conv_implicit_gemm_coopmat1_accf32_nhwc.glsl
// compiled with KATAGO_VULKAN_ACCUMULATE_FP16=1)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(conv5x5NhwcCoopmat1AccF16, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_FULL)
// Winograd coopmat1 GEMM, FP16 accumulation (winograd_gemm_coopmat1_accf16.glsl)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(coopmat1AccF16, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_FULL)
// NHWC 1x1 GEMM coopmat1 (gemm_strided_coopmat1_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(nhwcStridedCoopmat1, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16)
// NHWC 1x1 GEMM coopmat1, FP16 accumulation (gemm_strided_coopmat1_accf16_nhwc.glsl)
VULKAN_TUNE_COOPMAT1_TILE_GROUP(nhwcStridedCoopmat1AccF16, 128, 64, 64, 16, 32, 32, 16, 16, 16, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_FULL)
// NHWC 1x1 GEMM coopmat2 (gemm_strided_coopmat2_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(nhwcStridedCoopmat2, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16)
// NHWC 1x1 GEMM coopmat2, FP16 accumulation (gemm_strided_coopmat2_accf16_nhwc.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(nhwcStridedCoopmat2AccF16, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_FULL)
// Policy-independent weighted model times for the NHWC 1x1 GEMM variants
// (gemm_strided_coopmat{1,2}_accf{32,16}_nhwc.glsl,
// gemm_strided_dot2_accf{32,16}_nhwc.glsl). Zero means not measured.
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmCoopmat1F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_TIMEUS)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmCoopmat2F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmCoopmat1F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmCoopmat2F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_FULL)
// NHWC 1x1 GEMM DOT2 (gemm_strided_dot2_accf32_nhwc.glsl)
VULKAN_TUNE_DOT2_TILE_GROUP(nhwcStridedDot2, 128, 64, 64, 32, 32, 2, 4, 2, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_FULL)
// NHWC 1x1 GEMM DOT2, FP16 accumulation (gemm_strided_dot2_accf16_nhwc.glsl)
VULKAN_TUNE_DOT2_TILE_GROUP(nhwcStridedDot2AccF16, 128, 64, 64, 32, 32, 2, 4, 2, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_FULL)
// Policy-independent weighted model times (zero means not measured).
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmDot2F32TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, nhwcGemmDot2F16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2)
// NHWC 3x3 implicit GEMM coopmat2 (conv_implicit_gemm_coopmat2_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(conv3x3NhwcCoopmat2, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv3x3NhwcCoopmat1TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_TIMEUS)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv3x3NhwcCoopmat2TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2)
// NHWC 5x5 implicit GEMM coopmat2 (conv_implicit_gemm_coopmat2_accf32_nhwc.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(conv5x5NhwcCoopmat2, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv5x5NhwcCoopmat1TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv5x5NhwcCoopmat2TimeUs, 0, ())
#endif

// FP16-accumulation counterparts of the four conv TimeUs fields above. Without
// these the implicit-GEMM conv path had no way to compare accF16 against accF32
// and could only select accF16 when the accF32 variant was entirely absent, so
// the accF16 conv shaders were effectively unreachable on any device supporting
// accF32 coopmat. With them, conv3x3/5x5 goes through the same
// selectVariantTier policy as every other family.
#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv3x3NhwcCoopmat1AccF16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv3x3NhwcCoopmat2AccF16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv5x5NhwcCoopmat1AccF16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, conv5x5NhwcCoopmat2AccF16TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_FULL)
// NHWC 3x3 implicit GEMM coopmat2, FP16 accumulation
// (conv_implicit_gemm_coopmat2_accf32_nhwc.glsl compiled with
// KATAGO_VULKAN_ACCUMULATE_FP16=1)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(conv3x3NhwcCoopmat2AccF16, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_FULL)
// NHWC 5x5 implicit GEMM coopmat2, FP16 accumulation
// (conv_implicit_gemm_coopmat2_accf32_nhwc.glsl compiled with
// KATAGO_VULKAN_ACCUMULATE_FP16=1)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(conv5x5NhwcCoopmat2AccF16, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_FULL)
// Winograd coopmat2 GEMM (winograd_gemm_coopmat2_accf32.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(coopmat2, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16) || defined(VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_FULL)
// Winograd coopmat2 GEMM, FP16 accumulation (winograd_gemm_coopmat2_accf16.glsl)
VULKAN_TUNE_COOPMAT2_TILE_GROUP(coopmat2AccF16, 128, 64, 64, 32)
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_GPOOL_NHWC)
// GPool mask reduction (gpool_mask_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, gpoolNhwcXystride, 64, (16, 32, 64, 128))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_VALUE_HEAD_POOL_NHWC)
// Value-head pooling (value_head_pool_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, valueHeadPoolNhwcXystride, 64, (16, 32, 64, 128))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_SPATIAL_RMSNORM_NHWC)
// Spatial RMSNorm (transformer_spatial_rmsnorm_nhwc_pass1.glsl,
// transformer_spatial_rmsnorm_pass2.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, spatialRMSNormNhwcTile, 64, (32, 64, 128, 256))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_SWIGLU)
// SwiGLU (transformer_swiglu.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, swiGLULocalSizeX, 64, (64, 128, 256, 512))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_TILED)
// Tiled NHWC attention (transformer_attention_tiled_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcBlockQ, 32, (16, 32, 64))
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcBlockKV, 32, (16, 32, 64))
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcQPerThread, 1, (1, 2, 4))
#endif

#if defined(VULKAN_TUNE_PARAM_ALL)
// Policy-independent weighted model time (zero means not measured).
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_FULL)
// NHWC attention coopmat1 (transformer_attention_coopmat1_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1WorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1BlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1BlockKV, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1TM, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1TN, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1TK, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SubgroupSize, 32, ())
// Direct global K/V loads (0 or 1)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1DirectKV, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_SPLITK)
// Split-K attention (transformer_attention_coopmat1_split_k_nhwc.glsl,
// transformer_attention_split_k_resolve_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKWorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKBlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKBlockKV, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKTM, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKTN, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKTK, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKSubgroupSize, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKDirectKV, 0, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKKVChunkCount, 1, ())
// Split-K batch cutoff (0 disables)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat1SplitKCutoffBatch, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_TIMEUS)
// Attention coopmat1 maintenance1 (transformer_attention_coopmat1_maintenance1_nhwc.glsl,
// split-K: transformer_attention_coopmat1_maintenance1_split_k_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1WorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1BlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1BlockKV, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1TM, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1TN, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1TK, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SubgroupSize, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1DirectKV, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_SPLITK)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKWorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKBlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKBlockKV, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKTM, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKTN, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKTK, 16, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKSubgroupSize, 32, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKDirectKV, 0, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKKVChunkCount, 1, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmatMaintenance1SplitKCutoffBatch, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_FULL)
// NHWC attention coopmat2 (transformer_attention_coopmat2_accf32_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat2TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_FULL)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat2WorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat2BlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcCoopmat2BlockKV, 32, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_TIMEUS) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_FULL)
// NHWC attention DOT2 (transformer_attention_dot2_accf32_nhwc.glsl)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcDot2TimeUs, 0, ())
#endif

#if defined(VULKAN_TUNE_PARAM_ALL) || defined(VULKAN_TUNE_PARAM_ALL_ACCELERATED) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_FULL) || defined(VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2)
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcDot2WorkgroupSize, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcDot2BlockQ, 128, ())
VULKAN_TUNE_PARAM_FIELD(int32_t, attnNhwcDot2BlockKV, 32, ())
#endif

// ---------------------------------------------------------------------------
// Cleanup. Every trigger (and the VULKAN_TUNE_PARAM_FIELD body macro) is
// #undef'd here so the next include site in the same TU starts fresh.
// ---------------------------------------------------------------------------
#undef VULKAN_TUNE_PARAM_ALL
#undef VULKAN_TUNE_PARAM_ALL_ACCELERATED
#undef VULKAN_TUNE_PARAM_FIELD
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_SPLITK
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_SPLITK
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_ATTN_TILED
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_DIRECT
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_TILED
#undef VULKAN_TUNE_PARAM_KERNEL_GPOOL_NHWC
#undef VULKAN_TUNE_PARAM_KERNEL_NCHW_TO_NHWC
#undef VULKAN_TUNE_PARAM_KERNEL_SPATIAL_RMSNORM_NHWC
#undef VULKAN_TUNE_PARAM_KERNEL_SWIGLU
#undef VULKAN_TUNE_PARAM_KERNEL_VALUE_HEAD_POOL_NHWC
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_3X3_OUT_TILE
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_FULL
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_TIMEUS
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_TILED
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_TRANSFORM_NHWC
#undef VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_UNTRANSFORM_NHWC

// The tile-group shorthands are re-defined on every include, so drop them too.
#undef VULKAN_TUNE_COOPMAT1_TILE_GROUP
#undef VULKAN_TUNE_COOPMAT2_TILE_GROUP
#undef VULKAN_TUNE_DOT2_TILE_GROUP

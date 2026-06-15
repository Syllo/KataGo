#ifndef NEURALNET_VULKAN_TUNER_H_
#define NEURALNET_VULKAN_TUNER_H_

#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <vector>
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"

struct VulkanDeviceInfo;
struct CoopmatShape;  // vulkanhelpers.h
class Logger;

// VulkanTuneParams: all tuning knobs for the Vulkan backend in one place.
struct VulkanTuneParams {
  // Tunable fields (see vulkantuneparams_fields.inc for the single source of truth).
  // They are all int32_t except the structural string/masks, so a single
  // `int32_t VulkanTuneParams::*` pointer-to-member can address the tunable ones,
  // letting a kernel access and modify its own parameters.
#define VULKAN_TUNE_PARAM_ALL
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) TYPE NAME = DEFAULT;
#include "vulkantuneparams_fields.inc"

  bool isValid() const;

  // Kernel-flavor validity helpers over tunedKernelMask. Each flag is a
  // VulkanTuner::TunedKernel bit; hasKernelTuned reports whether a single
  // kernel/variant has been tuned, allOfKernelTuned/anyOfKernelTuned report
  // whether all/any listed kernels have been tuned, and
  // markKernelTuned/clearKernelTuned set and clear the listed bits.
  bool hasKernelTuned(int64_t flag) const;
  bool allOfKernelTuned(std::initializer_list<int64_t> flags) const;
  bool anyOfKernelTuned(std::initializer_list<int64_t> flags) const;
  void markKernelTuned(std::initializer_list<int64_t> flags);
  void clearKernelTuned(std::initializer_list<int64_t> flags);
  static void save(const std::string& filename, const VulkanTuneParams& config);
  static VulkanTuneParams load(const std::string& filename);
};

// VulkanTuner: per-GPU autotuning of the runtime-tunable VulkanTuneParams.
//
// This header holds only the Vulkan-API-free pieces (so command/tune.cpp can
// include it without Vulkan headers): the tune params, model info, device
// listing, file naming, and the command entry point. The actual tuning
// machinery (the TunableKernel interface in vulkankernels.h, the TuningContext
// device session in vulkanbackend.h, and the per-kernel benchmarks in
// vulkankernels.cpp) is free to depend on the Vulkan API.

namespace VulkanTuner {

  // Validity bit for each tunable kernel/variant. One bit per kernel flavor;
  // family sets are composed from these via the *FlavorBits() helpers below.
  enum TunedKernel : int64_t {
    TUNED_GEMM_DIRECT = 1LL << 0,
    TUNED_NCHW_TO_NHWC = 1LL << 1,
    TUNED_GPOOL_NHWC = 1LL << 2,
    TUNED_VALUE_HEAD_POOL_NHWC = 1LL << 3,
    TUNED_SPATIAL_RMSNORM_NHWC = 1LL << 4,

    TUNED_GEMM_STRIDED_TILED = 1LL << 5,
    TUNED_GEMM_STRIDED_DOT2 = 1LL << 6,
    TUNED_GEMM_STRIDED_DOT2_ACCF16 = 1LL << 7,
    TUNED_GEMM_STRIDED_COOPMAT1 = 1LL << 8,
    TUNED_GEMM_STRIDED_COOPMAT1_ACCF16 = 1LL << 9,
    TUNED_GEMM_STRIDED_COOPMAT2 = 1LL << 10,
    TUNED_GEMM_STRIDED_COOPMAT2_ACCF16 = 1LL << 11,

    TUNED_WINOGRAD_TILED = 1LL << 12,
    TUNED_WINOGRAD_DOT2 = 1LL << 13,
    TUNED_WINOGRAD_DOT2_ACCF16 = 1LL << 14,
    TUNED_WINOGRAD_COOPMAT1 = 1LL << 15,
    TUNED_WINOGRAD_COOPMAT1_ACCF16 = 1LL << 16,
    TUNED_WINOGRAD_COOPMAT2 = 1LL << 17,
    TUNED_WINOGRAD_COOPMAT2_ACCF16 = 1LL << 18,

    TUNED_CONV3X3_COOPMAT1 = 1LL << 19,
    TUNED_CONV3X3_COOPMAT1_ACCF16 = 1LL << 20,
    TUNED_CONV3X3_COOPMAT2 = 1LL << 21,
    TUNED_CONV3X3_COOPMAT2_ACCF16 = 1LL << 22,
    TUNED_CONV3X3_WINOGRAD = 1LL << 23,

    TUNED_CONV5X5_COOPMAT1 = 1LL << 24,
    TUNED_CONV5X5_COOPMAT1_ACCF16 = 1LL << 25,
    TUNED_CONV5X5_COOPMAT2 = 1LL << 26,
    TUNED_CONV5X5_COOPMAT2_ACCF16 = 1LL << 27,
    TUNED_CONV5X5_WINOGRAD = 1LL << 28,

    TUNED_ATTN_TILED = 1LL << 29,
    TUNED_ATTN_COOPMAT1 = 1LL << 30,
    TUNED_ATTN_COOPMAT1_SPLITK = 1LL << 31,
    TUNED_ATTN_MAINTENANCE1 = 1LL << 32,
    TUNED_ATTN_MAINTENANCE1_SPLITK = 1LL << 33,
    TUNED_ATTN_COOPMAT2 = 1LL << 34,
    TUNED_ATTN_DOT2 = 1LL << 35,

    TUNED_SWIGLU = 1LL << 36,

    // Unlike every other tier their validity is the Winograd layout signature
    // (nhwcWinogradTransformTunedLayout), so these bits mean "tuned for the
    // layout the signature currently records".
    TUNED_WINOGRAD_TRANSFORM = 1LL << 37,
    TUNED_WINOGRAD_UNTRANSFORM = 1LL << 38,
  };

  // Composed family sets over the per-kernel validity bits.
  inline constexpr int64_t stridedGemmFlavorBits() {
    return TUNED_GEMM_STRIDED_TILED | TUNED_GEMM_STRIDED_DOT2 | TUNED_GEMM_STRIDED_DOT2_ACCF16 |
           TUNED_GEMM_STRIDED_COOPMAT1 | TUNED_GEMM_STRIDED_COOPMAT1_ACCF16 | TUNED_GEMM_STRIDED_COOPMAT2 |
           TUNED_GEMM_STRIDED_COOPMAT2_ACCF16;
  }
  inline constexpr int64_t winogradGemmFlavorBits() {
    return TUNED_WINOGRAD_TILED | TUNED_WINOGRAD_DOT2 | TUNED_WINOGRAD_DOT2_ACCF16 | TUNED_WINOGRAD_COOPMAT1 |
           TUNED_WINOGRAD_COOPMAT1_ACCF16 | TUNED_WINOGRAD_COOPMAT2 | TUNED_WINOGRAD_COOPMAT2_ACCF16;
  }
  inline constexpr int64_t conv3x3FlavorBits() {
    return TUNED_CONV3X3_COOPMAT1 | TUNED_CONV3X3_COOPMAT1_ACCF16 | TUNED_CONV3X3_COOPMAT2 |
           TUNED_CONV3X3_COOPMAT2_ACCF16 | TUNED_CONV3X3_WINOGRAD;
  }
  inline constexpr int64_t conv5x5FlavorBits() {
    return TUNED_CONV5X5_COOPMAT1 | TUNED_CONV5X5_COOPMAT1_ACCF16 | TUNED_CONV5X5_COOPMAT2 |
           TUNED_CONV5X5_COOPMAT2_ACCF16 | TUNED_CONV5X5_WINOGRAD;
  }
  inline constexpr int64_t attentionVariantBits() {
    return TUNED_ATTN_COOPMAT1 | TUNED_ATTN_COOPMAT1_SPLITK | TUNED_ATTN_MAINTENANCE1 | TUNED_ATTN_MAINTENANCE1_SPLITK |
           TUNED_ATTN_COOPMAT2 | TUNED_ATTN_DOT2;
  }
  // Per-accelerator-variant bitsets, mapping the KATAGO_VULKAN_ACCEL_VARIANTS
  // filter families onto the tuned-kernel bits.
  inline constexpr int64_t coopmat1AccF32Bits() {
    return TUNED_WINOGRAD_COOPMAT1 | TUNED_GEMM_STRIDED_COOPMAT1 | TUNED_CONV3X3_COOPMAT1 | TUNED_CONV5X5_COOPMAT1 |
           TUNED_ATTN_COOPMAT1 | TUNED_ATTN_COOPMAT1_SPLITK | TUNED_ATTN_MAINTENANCE1 | TUNED_ATTN_MAINTENANCE1_SPLITK;
  }
  inline constexpr int64_t coopmat1AccF16Bits() {
    return TUNED_WINOGRAD_COOPMAT1_ACCF16 | TUNED_GEMM_STRIDED_COOPMAT1_ACCF16 | TUNED_CONV3X3_COOPMAT1_ACCF16 |
           TUNED_CONV5X5_COOPMAT1_ACCF16;
  }
  inline constexpr int64_t coopmat2AccF32Bits() {
    return TUNED_WINOGRAD_COOPMAT2 | TUNED_GEMM_STRIDED_COOPMAT2 | TUNED_CONV3X3_COOPMAT2 | TUNED_CONV5X5_COOPMAT2 |
           TUNED_ATTN_COOPMAT2;
  }
  inline constexpr int64_t coopmat2AccF16Bits() {
    return TUNED_WINOGRAD_COOPMAT2_ACCF16 | TUNED_GEMM_STRIDED_COOPMAT2_ACCF16 | TUNED_CONV3X3_COOPMAT2_ACCF16 |
           TUNED_CONV5X5_COOPMAT2_ACCF16;
  }
  inline constexpr int64_t dot2AccF32Bits() {
    return TUNED_WINOGRAD_DOT2 | TUNED_GEMM_STRIDED_DOT2 | TUNED_ATTN_DOT2;
  }
  inline constexpr int64_t dot2AccF16Bits() {
    return TUNED_WINOGRAD_DOT2_ACCF16 | TUNED_GEMM_STRIDED_DOT2_ACCF16;
  }
  inline constexpr int64_t allAccelVariantBits() {
    return coopmat1AccF32Bits() | coopmat1AccF16Bits() | coopmat2AccF32Bits() | coopmat2AccF16Bits() |
           dot2AccF32Bits() | dot2AccF16Bits();
  }

  struct TuningContext;

  // Shape of the NHWC Winograd transform/untransform data layout implied by the
  // currently-selected Winograd GEMM variant (alignment + packed tile dims + A
  // packing). Shared between the reconcile load-check in vulkantuner.cpp and the
  // transform benches in vulkantunebench.cpp; makeWinogradTransformBenchLayout
  // (vulkantuner.cpp) derives one from params/device.
  struct WinogradTransformBenchLayout {
    int mAlignment;
    int nAlignment;
    int kAlignment;
    int packedBM;
    int packedBK;
    int packedAPadWords;
  };

  // Shared bench-support helpers defined in vulkantuner.cpp and used both there
  // (reconcile/load checks) and by the VulkanKernels::*::bench implementations
  // in vulkantunebench.cpp.
  bool coopmatShapeIsSupported(const std::vector<CoopmatShape>& shapes, int32_t tm, int32_t tn, int32_t tk);
  WinogradTransformBenchLayout makeWinogradTransformBenchLayout(
    const VulkanTuneParams& cfg,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute);

  constexpr int DEFAULT_X_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_Y_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_BATCH_SIZE = 8;
  constexpr int DEFAULT_WINOGRAD_3X3_TILE_SIZE = 2;

  // ---- File path / naming conventions -------------------------------------

  // Always creates the tuning directory (both callers write files into it).
  std::string defaultDirectory(const std::string& homeDataDirOverride);
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

  // ---- Tune-file validity checks ------------------------------------------
  //
  // At startup we load a VulkanTuneParams file (explicit --vulkan-tuner-file,
  // or the per-(device,board-size,model) cache). These helpers decide whether
  // that file is still valid for the current GPU + model + fp16 settings, and
  // if not, which subset of kernels needs re-benchmarking. They are the glue
  // between vulkanbackend.cpp (which consumes tune files) and the re-tune
  // drivers below.

  // Winograd transform/untransform kernels pack activations into a data layout
  // whose shape depends on the currently-selected Winograd GEMM variant (its
  // tile sizes and packing). Whenever that GEMM layout changes, the transforms
  // must be re-tuned even though all other kernels stay valid. These helpers
  // fingerprint that layout (see the file-local
  // winogradTransformLayoutSignature in vulkantuner.cpp) so a stale tune file
  // can be detected.
  // True iff the transforms were tuned for the layout currently implied by
  // config/deviceInfo/fp16 (i.e. the recorded layout matches the current
  // layout signature).
  bool nhwcWinogradTransformsTunedForCurrentLayout(
    const VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute);
  // Bitmask of every TUNED_* kernel the given model actually needs on the
  // given device: walks the model's conv sizes (3x3/5x5/1x1) and transformer
  // presence, then intersects with the device's supported accelerator families
  // (coopmat1/2, dot2) and fp16 to keep only the tiers that are actually
  // selectable at runtime. Also ANDs out deviceInfo.disabledAccelVariantMask,
  // so filter-disabled families are not required (which is what makes the
  // tiled/winograd fallback become required when every accelerator family is
  // disabled). `full` additionally requires the non-accelerated fallback tiers
  // (used by `katago tuner --full`). Returns 0 for a null model.
  int64_t requiredKernelMask(
    const ModelDesc* modelDesc,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute,
    bool full);
  // Reconcile a loaded VulkanTuneParams against the current device and kernel
  // support. This is the single load-time validity gate: every field and tuned
  // entry is checked against the kernels' structural requirements and the
  // device's reported capabilities (coopmat shapes / subgroup size, flex-shape
  // granularity, shared-memory and workgroup limits, fp16/dot2/coopmat
  // support). Anything not currently supported is reset to its default value
  // and its tuned bit cleared; derived entries (the Winograd conv path and the
  // transform-layout signature) are cleared when the selected Winograd GEMM
  // variant becomes invalid. Variants disabled via
  // deviceInfo.disabledAccelVariantMask are left untouched (hidden during
  // validation, restored at the end). The invalid bits are derived from the
  // TimeUs fields (computeInvalidVariantBitsFromTimeUs) and the invariant
  // invalidKernelMask subset of tunedKernelMask is enforced at the end. Returns
  // true if anything was invalidated. After this function and a tuner run to
  // fill whatever was cleared, the params can be trusted to be valid for the
  // device.
  bool reconcileTuneParamsForDevice(
    VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    Logger* logger,
    bool fp16Storage,
    bool fp16Compute);
  // ---- Command-line entry point -------------------------------------------

  // Implements the `katago tuner` subcommand for the Vulkan backend (arg parsing,
  // device enumeration, per-GPU tune + save). Defined in vulkantuner.cpp so all
  // Vulkan tuning logic lives together; command/tune.cpp just forwards to it.
  int runTuneCommand(const std::vector<std::string>& args);
}  // namespace VulkanTuner

#endif  // NEURALNET_VULKAN_TUNER_H_

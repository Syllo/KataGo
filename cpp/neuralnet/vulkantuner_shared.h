#ifndef NEURALNET_VULKAN_TUNER_SHARED_H_
#define NEURALNET_VULKAN_TUNER_SHARED_H_

// Internal helpers shared between the tuner TUs (vulkantuner.cpp, the tuning
// machinery, and vulkantuneparams.cpp, the reconcile/load/save/command logic).
// This is NOT the API-free public header (vulkantuner.h): it pulls the Vulkan
// types (TuningContext, VulkanDeviceInfo, TunedCandidate) these helpers need,
// so only Vulkan-enabled TUs include it.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkankernels.h"

namespace VulkanTuner {

  // Time-budget for all kernel bench runs: probe a few iterations to estimate
  // per-dispatch cost, then clamp the timed run so each candidate takes at most
  // benchTargetSeconds wall-clock. Broad multi-shape sweeps multiply the budget
  // by every candidate and case, so they screen with the shorter sample below
  // and only confirm the finalists at the full budget.
  constexpr double BENCH_SCREEN_TARGET_SECONDS = 0.2;

  class ScopedTuningBenchTarget {
   public:
    ScopedTuningBenchTarget(TuningContext& ctx_, double target) : ctx(ctx_), oldTarget(ctx.benchTargetSeconds) {
      ctx.benchTargetSeconds = target;
    }
    ~ScopedTuningBenchTarget() { ctx.benchTargetSeconds = oldTarget; }

   private:
    TuningContext& ctx;
    double oldTarget;
  };

  // Round a wall-clock seconds measurement to a stored microsecond field,
  // clamped to the int32 range; 0 when the measurement is unusable.
  int32_t secondsToMicros(double seconds);

  // Self-apply one candidate's chosen (field, value) pairs to a config.
  void applyCandidate(const TunedCandidate& cand, VulkanTuneParams& cfg);

  // Compact string fingerprint of the NHWC Winograd transform/untransform data
  // layout (alignment + packed tile dims + fp16Storage) implied by the given
  // params/device. The transforms must be re-tuned whenever this signature
  // changes; two configs producing the same signature share tuned transforms.
  std::string winogradTransformLayoutSignature(
    const VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute);

  // Record the current layout signature in config, stamping it as tuned for
  // this layout. Called after a successful re-tune so future loads skip it.
  void markNhwcWinogradTransformsTunedForCurrentLayout(
    VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute);

  // Record "attempted but no valid candidate" for the accelerator gemm and
  // attention tiers: a tuned bit whose measured time score is missing. These
  // bits are recomputed from the TimeUs fields after tuning (and derived at
  // load time in reconcile), so invalidKernelMask stays consistent.
  int64_t computeInvalidVariantBitsFromTimeUs(const VulkanTuneParams& cfg);

  // ---- Coopmat2 (VK_NV_cooperative_matrix2) tile helpers ----
  // Shared between the candidate sweeps in vulkantuner.cpp and the reconcile
  // validation in vulkantuneparams.cpp (which gates on the family support flag
  // before delegating here).

  struct Coopmat2Tile {
    int32_t workgroupSize;
    int32_t bm;
    int32_t bn;
    int32_t bk;
  };

  inline bool coopmat2TileMatchesShape(const Coopmat2Tile& t, const Coopmat2FlexShape& s) {
    return t.workgroupSize == (int32_t)s.workgroupInvocations && t.bm % (int32_t)s.mGranularity == 0 &&
           t.bn % (int32_t)s.nGranularity == 0 && t.bk % (int32_t)s.kGranularity == 0;
  }

  // Whether the device reports a flex shape that `t` satisfies, within its
  // workgroup-invocation and max-flex-dimension limits. A non-positive tile is
  // never supported.
  //
  // Deliberately no shared-memory check here. The driver's workgroup-scope
  // reservation (coopmat2ReservedSharedBytes) is a physical-device constant, not
  // a function of the tile, and device init already refuses to report any
  // coopmat2 flex shape at all when it exceeds maxComputeSharedMemorySize (see
  // the requiredFeatures gate in queryDeviceInfo) -- so supportsCoopmat2F16 is
  // false and no tile ever reaches here. Callers whose *shader* stages its own
  // shared memory add reserved + their own sharedBytes() themselves; see
  // coopmat2ConvEligible.
  inline bool coopmat2TileSupported(
    const VulkanDeviceInfo& info,
    const std::vector<Coopmat2FlexShape>& shapes,
    const Coopmat2Tile& t) {
    if(t.bm <= 0 || t.bn <= 0 || t.bk <= 0 || t.workgroupSize <= 0)
      return false;
    if(info.coopmat2MaxWorkgroupSize > 0 && (uint32_t)t.workgroupSize > info.coopmat2MaxWorkgroupSize)
      return false;
    if(info.coopmat2MaxFlexDimension > 0) {
      uint32_t maxDim = std::max((uint32_t)t.bm, std::max((uint32_t)t.bn, (uint32_t)t.bk));
      if(maxDim > info.coopmat2MaxFlexDimension)
        return false;
    }
    for(const auto& s: shapes)
      if(coopmat2TileMatchesShape(t, s))
        return true;
    return false;
  }

  // ---- Tile -> VulkanTuneParams field mapping ----
  //
  // Every tile-shaped .inc field group lists its fields in the same order as the
  // corresponding tile struct members (verified against
  // vulkantuneparams_fields.inc). Shared by vulkantuner.cpp (applyTileFields zips a
  // group with a tile's scalars) and by the implicit-GEMM conv benches in
  // vulkantunebench.cpp (reading the is5x5-selected 3x3/5x5 tile). Keep each list
  // in tile-member order; the count static_asserts keep group size and tile member
  // count in step.

  // CoopmatTile member order: workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize.
  inline int32_t VulkanTuneParams::* const kWinogradCoopmat1TileFields[] = {
    &VulkanTuneParams::coopmat1WorkgroupSize,
    &VulkanTuneParams::coopmat1BM,
    &VulkanTuneParams::coopmat1BN,
    &VulkanTuneParams::coopmat1BK,
    &VulkanTuneParams::coopmat1SGM,
    &VulkanTuneParams::coopmat1SGN,
    &VulkanTuneParams::coopmat1TM,
    &VulkanTuneParams::coopmat1TN,
    &VulkanTuneParams::coopmat1TK,
    &VulkanTuneParams::coopmat1SubgroupSize};
  inline int32_t VulkanTuneParams::* const kWinogradCoopmat1AccF16TileFields[] = {
    &VulkanTuneParams::coopmat1AccF16WorkgroupSize,
    &VulkanTuneParams::coopmat1AccF16BM,
    &VulkanTuneParams::coopmat1AccF16BN,
    &VulkanTuneParams::coopmat1AccF16BK,
    &VulkanTuneParams::coopmat1AccF16SGM,
    &VulkanTuneParams::coopmat1AccF16SGN,
    &VulkanTuneParams::coopmat1AccF16TM,
    &VulkanTuneParams::coopmat1AccF16TN,
    &VulkanTuneParams::coopmat1AccF16TK,
    &VulkanTuneParams::coopmat1AccF16SubgroupSize};
  inline int32_t VulkanTuneParams::* const kConv3x3Coopmat1TileFields[] = {
    &VulkanTuneParams::conv3x3NhwcCoopmat1WorkgroupSize,
    &VulkanTuneParams::conv3x3NhwcCoopmat1BM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1BN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1BK,
    &VulkanTuneParams::conv3x3NhwcCoopmat1SGM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1SGN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1TM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1TN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1TK,
    &VulkanTuneParams::conv3x3NhwcCoopmat1SubgroupSize};
  inline int32_t VulkanTuneParams::* const kConv5x5Coopmat1TileFields[] = {
    &VulkanTuneParams::conv5x5NhwcCoopmat1WorkgroupSize,
    &VulkanTuneParams::conv5x5NhwcCoopmat1BM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1BN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1BK,
    &VulkanTuneParams::conv5x5NhwcCoopmat1SGM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1SGN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1TM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1TN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1TK,
    &VulkanTuneParams::conv5x5NhwcCoopmat1SubgroupSize};
  inline int32_t VulkanTuneParams::* const kConv3x3Coopmat1AccF16TileFields[] = {
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16WorkgroupSize,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16BM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16BN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16BK,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16SGM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16SGN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16TM,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16TN,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16TK,
    &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16SubgroupSize};
  inline int32_t VulkanTuneParams::* const kConv5x5Coopmat1AccF16TileFields[] = {
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16WorkgroupSize,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16BM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16BN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16BK,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16SGM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16SGN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16TM,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16TN,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16TK,
    &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16SubgroupSize};
  inline int32_t VulkanTuneParams::* const kNhwcStridedCoopmat1TileFields[] = {
    &VulkanTuneParams::nhwcStridedCoopmat1WorkgroupSize,
    &VulkanTuneParams::nhwcStridedCoopmat1BM,
    &VulkanTuneParams::nhwcStridedCoopmat1BN,
    &VulkanTuneParams::nhwcStridedCoopmat1BK,
    &VulkanTuneParams::nhwcStridedCoopmat1SGM,
    &VulkanTuneParams::nhwcStridedCoopmat1SGN,
    &VulkanTuneParams::nhwcStridedCoopmat1TM,
    &VulkanTuneParams::nhwcStridedCoopmat1TN,
    &VulkanTuneParams::nhwcStridedCoopmat1TK,
    &VulkanTuneParams::nhwcStridedCoopmat1SubgroupSize};
  inline int32_t VulkanTuneParams::* const kNhwcStridedCoopmat1AccF16TileFields[] = {
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16WorkgroupSize,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16BM,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16BN,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16BK,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16SGM,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16SGN,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16TM,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16TN,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16TK,
    &VulkanTuneParams::nhwcStridedCoopmat1AccF16SubgroupSize};
  // Coopmat2Tile member order: workgroupSize, bm, bn, bk.
  inline int32_t VulkanTuneParams::* const kWinogradCoopmat2TileFields[] = {
    &VulkanTuneParams::coopmat2WorkgroupSize,
    &VulkanTuneParams::coopmat2BM,
    &VulkanTuneParams::coopmat2BN,
    &VulkanTuneParams::coopmat2BK};
  inline int32_t VulkanTuneParams::* const kWinogradCoopmat2AccF16TileFields[] = {
    &VulkanTuneParams::coopmat2AccF16WorkgroupSize,
    &VulkanTuneParams::coopmat2AccF16BM,
    &VulkanTuneParams::coopmat2AccF16BN,
    &VulkanTuneParams::coopmat2AccF16BK};
  inline int32_t VulkanTuneParams::* const kNhwcStridedCoopmat2TileFields[] = {
    &VulkanTuneParams::nhwcStridedCoopmat2WorkgroupSize,
    &VulkanTuneParams::nhwcStridedCoopmat2BM,
    &VulkanTuneParams::nhwcStridedCoopmat2BN,
    &VulkanTuneParams::nhwcStridedCoopmat2BK};
  inline int32_t VulkanTuneParams::* const kNhwcStridedCoopmat2AccF16TileFields[] = {
    &VulkanTuneParams::nhwcStridedCoopmat2AccF16WorkgroupSize,
    &VulkanTuneParams::nhwcStridedCoopmat2AccF16BM,
    &VulkanTuneParams::nhwcStridedCoopmat2AccF16BN,
    &VulkanTuneParams::nhwcStridedCoopmat2AccF16BK};
  inline int32_t VulkanTuneParams::* const kConv3x3Coopmat2TileFields[] = {
    &VulkanTuneParams::conv3x3NhwcCoopmat2WorkgroupSize,
    &VulkanTuneParams::conv3x3NhwcCoopmat2BM,
    &VulkanTuneParams::conv3x3NhwcCoopmat2BN,
    &VulkanTuneParams::conv3x3NhwcCoopmat2BK};
  inline int32_t VulkanTuneParams::* const kConv5x5Coopmat2TileFields[] = {
    &VulkanTuneParams::conv5x5NhwcCoopmat2WorkgroupSize,
    &VulkanTuneParams::conv5x5NhwcCoopmat2BM,
    &VulkanTuneParams::conv5x5NhwcCoopmat2BN,
    &VulkanTuneParams::conv5x5NhwcCoopmat2BK};
  inline int32_t VulkanTuneParams::* const kConv3x3Coopmat2AccF16TileFields[] = {
    &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16WorkgroupSize,
    &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16BM,
    &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16BN,
    &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16BK};
  inline int32_t VulkanTuneParams::* const kConv5x5Coopmat2AccF16TileFields[] = {
    &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16WorkgroupSize,
    &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16BM,
    &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16BN,
    &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16BK};
  // Dot2Tile member order: workgroupSize, bm, bn, sgm, sgn, sgmIter, tm, tn, subgroupSize.
  inline int32_t VulkanTuneParams::* const kWinogradDot2TileFields[] = {
    &VulkanTuneParams::dot2WorkgroupSize,
    &VulkanTuneParams::dot2BM,
    &VulkanTuneParams::dot2BN,
    &VulkanTuneParams::dot2SGM,
    &VulkanTuneParams::dot2SGN,
    &VulkanTuneParams::dot2SGMIter,
    &VulkanTuneParams::dot2TM,
    &VulkanTuneParams::dot2TN,
    &VulkanTuneParams::dot2SubgroupSize};
  inline int32_t VulkanTuneParams::* const kWinogradDot2AccF16TileFields[] = {
    &VulkanTuneParams::dot2AccF16WorkgroupSize,
    &VulkanTuneParams::dot2AccF16BM,
    &VulkanTuneParams::dot2AccF16BN,
    &VulkanTuneParams::dot2AccF16SGM,
    &VulkanTuneParams::dot2AccF16SGN,
    &VulkanTuneParams::dot2AccF16SGMIter,
    &VulkanTuneParams::dot2AccF16TM,
    &VulkanTuneParams::dot2AccF16TN,
    &VulkanTuneParams::dot2AccF16SubgroupSize};
  inline int32_t VulkanTuneParams::* const kNhwcStridedDot2TileFields[] = {
    &VulkanTuneParams::nhwcStridedDot2WorkgroupSize,
    &VulkanTuneParams::nhwcStridedDot2BM,
    &VulkanTuneParams::nhwcStridedDot2BN,
    &VulkanTuneParams::nhwcStridedDot2SGM,
    &VulkanTuneParams::nhwcStridedDot2SGN,
    &VulkanTuneParams::nhwcStridedDot2SGMIter,
    &VulkanTuneParams::nhwcStridedDot2TM,
    &VulkanTuneParams::nhwcStridedDot2TN,
    &VulkanTuneParams::nhwcStridedDot2SubgroupSize};
  inline int32_t VulkanTuneParams::* const kNhwcStridedDot2AccF16TileFields[] = {
    &VulkanTuneParams::nhwcStridedDot2AccF16WorkgroupSize,
    &VulkanTuneParams::nhwcStridedDot2AccF16BM,
    &VulkanTuneParams::nhwcStridedDot2AccF16BN,
    &VulkanTuneParams::nhwcStridedDot2AccF16SGM,
    &VulkanTuneParams::nhwcStridedDot2AccF16SGN,
    &VulkanTuneParams::nhwcStridedDot2AccF16SGMIter,
    &VulkanTuneParams::nhwcStridedDot2AccF16TM,
    &VulkanTuneParams::nhwcStridedDot2AccF16TN,
    &VulkanTuneParams::nhwcStridedDot2AccF16SubgroupSize};

  // Every group must hold exactly one entry per tile member.
  static_assert(
    std::size(kWinogradCoopmat1TileFields) == 10 && std::size(kWinogradCoopmat1AccF16TileFields) == 10 &&
      std::size(kConv3x3Coopmat1TileFields) == 10 && std::size(kConv5x5Coopmat1TileFields) == 10 &&
      std::size(kConv3x3Coopmat1AccF16TileFields) == 10 && std::size(kConv5x5Coopmat1AccF16TileFields) == 10 &&
      std::size(kNhwcStridedCoopmat1TileFields) == 10 && std::size(kNhwcStridedCoopmat1AccF16TileFields) == 10,
    "coopmat1 tile field groups must hold 10 fields (CoopmatTile members)");
  static_assert(
    std::size(kWinogradCoopmat2TileFields) == 4 && std::size(kWinogradCoopmat2AccF16TileFields) == 4 &&
      std::size(kNhwcStridedCoopmat2TileFields) == 4 && std::size(kNhwcStridedCoopmat2AccF16TileFields) == 4 &&
      std::size(kConv3x3Coopmat2TileFields) == 4 && std::size(kConv5x5Coopmat2TileFields) == 4 &&
      std::size(kConv3x3Coopmat2AccF16TileFields) == 4 && std::size(kConv5x5Coopmat2AccF16TileFields) == 4,
    "coopmat2 tile field groups must hold 4 fields (Coopmat2Tile members)");
  static_assert(
    std::size(kWinogradDot2TileFields) == 9 && std::size(kWinogradDot2AccF16TileFields) == 9 &&
      std::size(kNhwcStridedDot2TileFields) == 9 && std::size(kNhwcStridedDot2AccF16TileFields) == 9,
    "dot2 tile field groups must hold 9 fields (Dot2Tile members)");
  // Full structural equality over every VulkanTuneParams field.
  bool operator==(const VulkanTuneParams& a, const VulkanTuneParams& b);

  // True if the model has a transformer attention block anywhere in its trunk
  // (including inside nested bottleneck blocks).
  bool modelHasTransformer(const ModelDesc* desc);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKAN_TUNER_SHARED_H_

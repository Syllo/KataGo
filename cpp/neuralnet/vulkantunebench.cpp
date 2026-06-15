#ifdef USE_VULKAN_BACKEND

// VulkanKernels member impls that need VulkanTuner::TuningContext (a full type only
// available from vulkanbackend.h). Each instantiable kernel (TunableKernel subclass)
// defines its bench() here; keeping them out of vulkantuner.cpp leaves that file
// focused on the tuning machinery.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "../core/test.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkankernels.h"
#include "../neuralnet/vulkantuner_shared.h"

using namespace std;
using namespace VulkanHelpers;

namespace VulkanTuner {

  // Deterministic pseudo-random fill in [-0.5, 0.5). Seeded from problem dims so
  // a candidate's output is reproducible across configs (RMSE comparison is valid).
  static void fillRandom(vector<float>& v, uint32_t seed) {
    for(auto& x: v) {
      seed = seed * 1664525u + 1013904223u;
      x = ((seed >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
  }

  constexpr int BENCH_WARMUP_CHUNK_SIZE = 64;
  constexpr int BENCH_WARMUP_CHUNKS = 1;
  constexpr int BENCH_PROBE_ITERS = 5;

  // Run a time-budgeted bench: first do one fixed untimed warmup chunk, then
  // probe BENCH_PROBE_ITERS dispatches to estimate per-iteration cost and time
  // the clamped run. Returns {totalSeconds, effectiveIters}, or {-1, 0} on failure.
  struct BenchResult {
    double seconds;
    int iters;
  };
  static BenchResult
  timeBudgetedBench(VulkanTuner::TuningContext& s, int maxIters, const std::function<void()>& recordOne) {
    s.warmupDispatches(BENCH_WARMUP_CHUNK_SIZE, BENCH_WARMUP_CHUNKS, recordOne);
    double probeSeconds = s.timeDispatches(BENCH_PROBE_ITERS, recordOne);
    if(probeSeconds <= 0.0)
      return {-1.0, 0};
    int budgetIters = (int)(s.benchTargetSeconds / (probeSeconds / BENCH_PROBE_ITERS));
    int effectiveIters = std::clamp(budgetIters, BENCH_PROBE_ITERS, maxIters);
    double seconds = s.timeDispatches(effectiveIters, recordOne);
    return {seconds, effectiveIters};
  }

}  // namespace VulkanTuner

using namespace VulkanTuner;

namespace VulkanKernels {

  KernelBench SwiGLU::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1 || !isConfigSupported(cfg.swiGLULocalSizeX))
      return result;
    try {
      const size_t elts = (size_t)problemBatchSize * problemChannels * problemXySize;
      if(elts == 0 || elts % ELEMENTS_PER_VEC != 0)
        return result;
      std::vector<float> mainData(elts), gateData(elts);
      VulkanTuner::fillRandom(mainData, (uint32_t)(problemChannels * 73856093u ^ problemXySize * 19349663u));
      VulkanTuner::fillRandom(gateData, (uint32_t)(problemChannels * 83492791u ^ problemBatchSize * 2654435761u));
      VBuf mainBuf = s.makeInputBufFP(mainData);
      VBuf gateBuf = s.makeInputBufFP(gateData);
      ComputeKernel kernel = build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg);
      PC pc = {(int)elts};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(cctx, kernel, mainBuf.get(), gateBuf.get(), mainBuf.get(), pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, mainBuf->buffer);
      };
      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds > 0.0) {
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
      }
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: swiGLU candidate failed: ") + e.what());
      return result;
    }
  }

  namespace {
    void fillAttentionRopeTables(
      std::vector<float>& cosData,
      std::vector<float>& sinData,
      int tableHeads,
      int numPairs,
      int numTokens) {
      for(int h = 0; h < tableHeads; h++)
        for(int pair = 0; pair < numPairs; pair++)
          for(int pos = 0; pos < numTokens; pos++) {
            const size_t i = ((size_t)h * numPairs + pair) * numTokens + pos;
            const float angle = 0.013f * (float)(pair + 1) * (float)(pos + 1) + 0.071f * (float)h;
            cosData[i] = std::cos(angle);
            sinData[i] = std::sin(angle);
          }
    }

    void transposeAttentionRopeTables(
      std::vector<float>& cosData,
      std::vector<float>& sinData,
      int tableHeads,
      int numPairs,
      int numTokens) {
      auto transpose = [&](std::vector<float>& table) {
        std::vector<float> transposed(table.size());
        for(int h = 0; h < tableHeads; h++)
          for(int pair = 0; pair < numPairs; pair++)
            for(int pos = 0; pos < numTokens; pos++)
              transposed[((size_t)h * numTokens + pos) * numPairs + pair] =
                table[((size_t)h * numPairs + pair) * numTokens + pos];
        table.swap(transposed);
      };
      transpose(cosData);
      transpose(sinData);
    }
  }  // namespace

  namespace {

    // Run the timed loop, download the output, and finalize `result`; destroys
    // `kernel` on both the success and unusable-timing paths. `fp16` selects the
    // fp16-aware download (output stored as fp16) vs the plain fp32 one. Returns
    // false when the timed run was unusable (result stays empty).
    bool timeAndCollect(
      VulkanTuner::TuningContext& s,
      int iters,
      const std::function<void()>& recordOne,
      KernelBench& result,
      VulkanBuffer* outputBuf,
      size_t outputElts,
      bool fp16,
      ComputeKernel& kernel) {
      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return false;
      }
      result.output.resize(outputElts);
      if(fp16)
        s.downloadFloatsFP(outputBuf, result.output, outputElts);
      else
        s.downloadFloats(outputBuf, result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return true;
    }

    // Read the is5x5-selected 3x3/5x5 conv tile fields from the config into `out`
    // (tile-member order), via the shared k*TileFields tables.
    template<size_t N>
    static void convTileFromConfig(
      const VulkanTuneParams& cfg,
      bool is5x5,
      int32_t VulkanTuneParams::* const (&f3)[N],
      int32_t VulkanTuneParams::* const (&f5)[N],
      std::array<int32_t, N>& out) {
      for(size_t i = 0; i < N; i++)
        out[i] = cfg.*(is5x5 ? f5[i] : f3[i]);
    }

    // Shared problem parameters for the attention bench wrappers (all four read
    // the same member set: batchSize, tokens, head/vHead dims, heads, rope).
    struct AttentionBenchProblem {
      int batchSize, numTokens, validTokens, headDim, vHeadDim, numHeads, numKVHeads;
      bool useRope, learnableRope;
    };

    // q/k/v/mask (+ optional RoPE cos/sin) input buffers and the padded output
    // buffer for one attention bench problem. Deterministic fill so a
    // candidate's output is comparable across configs (RMSE vs the reference).
    struct AttentionBenchBuffers {
      size_t qElts, kElts, vElts, outElts, maskElts;
      VBuf q, k, v, mask, out, cos, sin;
    };

    AttentionBenchBuffers makeAttentionBenchBuffers(
      VulkanTuner::TuningContext& s,
      const AttentionBenchProblem& p,
      bool transposeRope,
      bool fp16Out) {
      AttentionBenchBuffers b;
      b.qElts = (size_t)p.batchSize * p.numHeads * p.headDim * p.numTokens;
      b.kElts = (size_t)p.batchSize * p.numKVHeads * p.headDim * p.numTokens;
      b.vElts = (size_t)p.batchSize * p.numKVHeads * p.vHeadDim * p.numTokens;
      b.outElts = (size_t)p.batchSize * p.numHeads * p.vHeadDim * p.numTokens;
      b.maskElts = (size_t)p.batchSize * p.numTokens;
      std::vector<float> qData(b.qElts), kData(b.kElts), vData(b.vElts), maskData(b.maskElts, 0.0f);
      for(int n = 0; n < p.batchSize; n++)
        for(int pos = 0; pos < p.validTokens; pos++)
          maskData[(size_t)n * p.numTokens + pos] = 1.0f;
      uint32_t seed = (uint32_t)(p.numTokens * 73856093u ^ p.headDim * 19349663u ^ p.numHeads * 83492791u);
      VulkanTuner::fillRandom(qData, seed);
      VulkanTuner::fillRandom(kData, seed ^ 0x1111u);
      VulkanTuner::fillRandom(vData, seed ^ 0x2222u);
      b.q = s.makeInputBufFP(qData);
      b.k = s.makeInputBufFP(kData);
      b.v = s.makeInputBufFP(vData);
      b.mask = s.makeInputBufFP(maskData);
      b.out = makeDeviceBuf(s.device(), s.memProps(), b.outElts, fp16Out);
      if(p.useRope) {
        const int numPairs = p.headDim / 2;
        const int tableHeads = p.learnableRope ? p.numKVHeads : 1;
        const size_t tableElts = (size_t)tableHeads * numPairs * (size_t)p.numTokens;
        std::vector<float> cosData(tableElts, 1.0f), sinData(tableElts, 0.0f);
        fillAttentionRopeTables(cosData, sinData, tableHeads, numPairs, p.numTokens);
        if(transposeRope)
          transposeAttentionRopeTables(cosData, sinData, tableHeads, numPairs, p.numTokens);
        b.cos = s.makeInputBuf(cosData);
        b.sin = s.makeInputBuf(sinData);
      }
      return b;
    }

    // Time the attention kernel, download the padded output, then zero the
    // padding tokens beyond validTokens so the RMSE comparison against the tiled
    // reference sees only real tokens. Returns false when the timed run failed.
    bool timeAttentionBench(
      VulkanTuner::TuningContext& s,
      int iters,
      const std::function<void()>& recordOne,
      const AttentionBenchBuffers& b,
      const AttentionBenchProblem& p,
      KernelBench& result,
      ComputeKernel& kernel) {
      if(!timeAndCollect(s, iters, recordOne, result, b.out.get(), b.outElts, true, kernel))
        return false;
      for(int n = 0; n < p.batchSize; n++)
        for(int pos = p.validTokens; pos < p.numTokens; pos++) {
          size_t base = ((size_t)n * p.numTokens + pos) * p.numHeads * p.vHeadDim;
          std::fill(result.output.begin() + base, result.output.begin() + base + (size_t)p.numHeads * p.vHeadDim, 0.0f);
        }
      return true;
    }
  }  // namespace

  namespace {

    template<typename BuildKernel, typename DispatchKernel>
    KernelBench benchWinogradGemmVariant(
      VulkanTuner::TuningContext& s,
      int iters,
      int problemM,
      int problemN,
      int problemK,
      int problemNumBatches,
      string_view candidateName,
      BuildKernel&& buildKernel,
      DispatchKernel&& dispatchKernel,
      bool packB = false,
      int packedBM = 0,
      int packedBN = 0,
      int packedBK = 0,
      int packedPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS) {
      KernelBench result;
      if(iters < 1)
        return result;
      try {
        ComputeKernel kernel = buildKernel();

        size_t aElts = (size_t)problemNumBatches * problemK * problemM;
        size_t bElts = (size_t)problemNumBatches * problemK * problemN;
        size_t cElts = (size_t)problemNumBatches * problemN * problemM;
        std::vector<float> aData(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aData, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);

        std::vector<float> aUploadData, bUploadData;
        aUploadData =
          packWinogradRowMajorAData(aData, problemNumBatches, problemM, problemK, packedBM, packedBK, packedPadWords);
        if(packB)
          bUploadData = packWinogradCoopmatBWeights(
            bData, problemNumBatches, problemN, problemK, packedBN, packedBK, packedPadWords);

        VBuf aBuf = s.makeInputBufFP(aUploadData);
        VBuf bBuf = s.makeInputBufFP(packB ? bUploadData : bData);
        VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, s.fp16Storage);

        // strideA is the row-major packed A tile stride in vec4 units.
        // strideB is variant-specific: normal Winograd GEMM uses K*(N/4), coopmat uses packed tile words.
        // strideC in vec4 units = (N/4)*M (M ÷4 by the bench roundUp to a ÷4 multiple).
        size_t strideA = winogradPackedRowMajorAStrideWords(problemM, problemK, packedBM, packedBK, packedPadWords);
        size_t strideB = packB
                           ? winogradCoopmatPackedBStrideWords(problemN, problemK, packedBN, packedBK, packedPadWords)
                           : (size_t)problemK * (size_t)(problemN / 4);
        testAssert(strideA <= (size_t)std::numeric_limits<int>::max());
        testAssert(strideB <= (size_t)std::numeric_limits<int>::max());
        WinogradGemm::PC pc = {
          problemM,
          problemN,
          (int)strideA,                // strideA, vec4
          (int)strideB,                // strideB, vec4
          (problemN / 4) * problemM};  // strideC, vec4

        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        auto recordOne = [&]() {
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemNumBatches);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
        result.output.resize(cElts);
        s.downloadFloatsFP(cBuf.get(), result.output, cElts);
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernel.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        // A lost device is not a bad candidate: it invalidates every later
        // bench too, so stop the sweep instead of logging thousands of
        // indistinguishable "candidate failed" lines.
        if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
          throw;
        if(s.logger != nullptr)
          s.logger->write(std::string("VulkanTuner: ") + string(candidateName) + " candidate failed: " + e.what());
        return result;
      }
    }

    // Build a canonical [batch, K, M] reference ordering, transpose only the uploaded A buffer to NHWC
    // [batch, M, K], then transpose C back before RMSE comparison. This times
    // exactly the native runtime kernel without letting layout change the
    // correctness oracle.
    //
    // packedBN==0 skips the B-packing step and uploads the plain [K, N] tile;
    // the tiled NHWC kernel consumes the plain K-major B layout.
    template<typename BuildKernel, typename DispatchKernel>
    KernelBench benchGemmStridedNhwcVariant(
      VulkanTuner::TuningContext& s,
      int iters,
      int problemBatchSize,
      int problemM,
      int problemN,
      int problemK,
      string_view candidateName,
      BuildKernel&& buildKernel,
      DispatchKernel&& dispatchKernel,
      int packedBN,
      int packedBK,
      int packedPadScalars,
      bool rowMajorPackedB = false) {
      KernelBench result;
      if(iters < 1)
        return result;
      try {
        ComputeKernel kernel = buildKernel();
        const size_t aElts = (size_t)problemBatchSize * problemK * problemM;
        const size_t bElts = (size_t)problemK * problemN;
        const size_t cElts = (size_t)problemBatchSize * problemN * problemM;
        std::vector<float> aCanonical(aElts), aNhwc(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aCanonical, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);
        for(int batch = 0; batch < problemBatchSize; batch++)
          for(int m = 0; m < problemM; m++)
            for(int k = 0; k < problemK; k++)
              aNhwc[((size_t)batch * problemM + m) * problemK + k] =
                aCanonical[((size_t)batch * problemK + k) * problemM + m];

        std::vector<float> bUploadData;
        if(packedBN > 0)
          bUploadData =
            rowMajorPackedB
              ? packStridedGemmBWeightsRowMajor(bData, 1, problemN, problemK, packedBN, packedBK, packedPadScalars)
              : packStridedGemmBWeights(bData, 1, problemN, problemK, packedBN, packedBK, packedPadScalars);
        VBuf aBuf = s.makeInputBufFP(aNhwc);
        VBuf bBuf = s.makeInputBufFP(packedBN > 0 ? bUploadData : bData);
        VBuf cBuf = s.makeInputBufFP(std::vector<float>(cElts, 0.0f));
        GemmStridedPC pc = {
          problemM,
          problemN,
          problemN,
          problemK * (problemM / 4),
          0,  // Runtime convolution weights are shared across the batch.
          problemN * (problemM / 4)};
        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemBatchSize);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };
        s.beginRecording();
        recordOne();
        s.submitAndWait();
        std::vector<float> cNhwc(cElts);
        s.downloadFloatsFP(cBuf.get(), cNhwc, cElts);
        result.output.resize(cElts);
        for(int batch = 0; batch < problemBatchSize; batch++)
          for(int m = 0; m < problemM; m++)
            for(int n = 0; n < problemN; n++)
              result.output[((size_t)batch * problemN + n) * problemM + m] =
                cNhwc[((size_t)batch * problemM + m) * problemN + n];
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernel.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        // A lost device is not a bad candidate: it invalidates every later
        // bench too, so stop the sweep instead of logging thousands of
        // indistinguishable "candidate failed" lines.
        if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
          throw;
        if(s.logger != nullptr)
          s.logger->write(std::string("VulkanTuner: ") + string(candidateName) + " candidate failed: " + e.what());
        return result;
      }
    }

  }  // namespace

  KernelBench WinogradGemm::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemm",
      [&]() { return WinogradGemm::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg, (uint32_t)problemK); },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemm::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.winogradGemmM,
      0,
      cfg.winogradGemmK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }

  namespace {
    KernelBench benchLayoutTransformDirection(
      VulkanTuner::TuningContext& s,
      const VulkanTuneParams& cfg,
      int iters,
      int problemBatchSize,
      int problemSpatialSize,
      const std::vector<int>& problemChannels) {
      KernelBench result;
      if(iters < 1 || problemBatchSize < 1 || problemSpatialSize < 1 || problemChannels.empty())
        return result;

      LayoutTransformKernels kernels;
      try {
        kernels = NchwToNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg);

        struct CaseBuffers {
          int channels;
          size_t elements;
          VBuf input;
          VBuf output;
        };
        std::vector<CaseBuffers> cases;
        cases.reserve(problemChannels.size());
        for(int channels: problemChannels) {
          const size_t elements = (size_t)problemBatchSize * (size_t)channels * (size_t)problemSpatialSize;
          std::vector<float> inputData(elements);
          VulkanTuner::fillRandom(
            inputData,
            (uint32_t)(channels * 73856093u ^ problemSpatialSize * 19349663u ^ problemBatchSize * 83492791u));
          cases.push_back(
            {channels,
             elements,
             s.makeInputBufFP(inputData),
             makeDeviceBuf(s.device(), s.memProps(), elements, s.fp16Storage)});
        }

        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          for(CaseBuffers& cs: cases) {
            NchwToNhwc::PC pc = {cs.channels, cs.channels, problemSpatialSize, problemBatchSize};
            NchwToNhwc::dispatch(cctx, kernels, cs.input.get(), cs.output.get(), pc);
            VulkanHelpers::cmdComputeBarrier(s.cmd, cs.output->buffer);
          }
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernels.destroy(s.device());
          return result;
        }

        size_t totalElements = 0;
        for(const CaseBuffers& cs: cases)
          totalElements += cs.elements;
        result.output.reserve(totalElements);
        for(CaseBuffers& cs: cases) {
          std::vector<float> output(cs.elements);
          s.downloadFloatsFP(cs.output.get(), output, cs.elements);
          result.output.insert(result.output.end(), output.begin(), output.end());
        }
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernels.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        kernels.destroy(s.device());
        // A lost device is not a bad candidate: it invalidates every later
        // bench too, so stop the sweep instead of logging thousands of
        // indistinguishable "candidate failed" lines.
        if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
          throw;
        if(s.logger != nullptr)
          s.logger->write(std::string("VulkanTuner: nchwToNhwc candidate failed: ") + e.what());
        return result;
      }
    }
  }  // namespace

  KernelBench NchwToNhwcTuner::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchLayoutTransformDirection(s, cfg, iters, problemBatchSize, problemSpatialSize, problemChannels);
  }

  KernelBench WinogradTransformNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = nhwcWinograd3x3OutTileFor(cfg);
      const int inTile = outTile + 2;
      int numTilesX = (problemNnXLen + outTile - 1) / outTile;
      int numTilesY = (problemNnYLen + outTile - 1) / outTile;
      int ntxty = problemBatchSize * numTilesX * numTilesY;
      WinogradTransformBenchLayout layout =
        makeWinogradTransformBenchLayout(cfg, s.dev->info, s.fp16Storage, s.fp16Compute);
      int numTilesPadded = ((ntxty + layout.mAlignment - 1) / layout.mAlignment) * layout.mAlignment;
      int numInChannelsPadded = ((problemInChannels + layout.kAlignment - 1) / layout.kAlignment) * layout.kAlignment;
      int paddedSpatialSize = problemNnXLen * problemNnYLen;

      size_t inputElts = (size_t)problemBatchSize * problemInChannels * paddedSpatialSize;
      size_t packedStrideA = winogradPackedRowMajorAStrideWords(
        numTilesPadded, numInChannelsPadded, layout.packedBM, layout.packedBK, layout.packedAPadWords);
      size_t outputElts = (size_t)inTile * inTile * packedStrideA * 4;

      std::vector<float> inputData(inputElts);
      VulkanTuner::fillRandom(
        inputData,
        (uint32_t)(problemNnXLen * 73856093u ^ problemInChannels * 19349663u ^ problemBatchSize * 83492791u ^ 41u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradTransformNhwc::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemInChannels,
        numInChannelsPadded,
        ntxty,
        numTilesPadded,
        paddedSpatialSize};

      ComputeKernel kernel = WinogradTransformNhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        cfg,
        3,
        -1,
        layout.packedBM,
        layout.packedBK,
        layout.packedAPadWords);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradTransformNhwc::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };
      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: winogradNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench WinogradUntransformNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = nhwcWinograd3x3OutTileFor(cfg);
      const int inTile = outTile + 2;
      int numTilesX = (problemNnXLen + outTile - 1) / outTile;
      int numTilesY = (problemNnYLen + outTile - 1) / outTile;
      int ntxty = problemBatchSize * numTilesX * numTilesY;
      WinogradTransformBenchLayout layout =
        makeWinogradTransformBenchLayout(cfg, s.dev->info, s.fp16Storage, s.fp16Compute);
      int numTilesPadded = ((ntxty + layout.mAlignment - 1) / layout.mAlignment) * layout.mAlignment;
      int numOutChannelsPadded = ((problemOutChannels + layout.nAlignment - 1) / layout.nAlignment) * layout.nAlignment;
      int paddedSpatialSize = problemNnXLen * problemNnYLen;

      size_t inputElts = (size_t)inTile * inTile * numOutChannelsPadded * numTilesPadded;
      size_t outputElts = (size_t)problemBatchSize * problemOutChannels * paddedSpatialSize;

      std::vector<float> inputData(inputElts);
      VulkanTuner::fillRandom(
        inputData,
        (uint32_t)(problemNnXLen * 73856093u ^ problemOutChannels * 19349663u ^ problemBatchSize * 83492791u ^ 53u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradUntransformNhwc::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemOutChannels,
        numOutChannelsPadded,
        numTilesPadded,
        paddedSpatialSize};

      // ocVec is derived from the unpadded problemOutChannels, matching the
      // real construction site in vulkanlayers.h -- padding is irrelevant to
      // whether the store can address 4 contiguous physical channels.
      const int ocVec = (problemOutChannels % 4 == 0) ? 4 : 1;
      ComputeKernel kernel =
        WinogradUntransformNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg, 3, false, ocVec);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradUntransformNhwc::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc, problemBatchSize, ocVec);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };
      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: winogradUntransformNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench GemmStridedTiledNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedTiledNhwc",
      [&]() { return GemmStridedTiledNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg, (uint32_t)problemK); },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiledNhwc::PC& pc,
        int batchSize) { GemmStridedTiledNhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      0,  // No B repack; the tiled NHWC shader consumes the canonical B layout.
      0,
      0);
  }

  KernelBench WinogradGemmDot2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16 || !s.fp16Storage)
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmDot2",
      [&]() {
        return WinogradGemmDot2::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg,
          problemK,
          s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize) ? s.dev->info.subgroupSize : 0u);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmDot2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.dot2BM,
      0,
      VulkanKernels::DOT2_BK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }
  KernelBench WinogradGemmDot2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmDot2AccF16",
      [&]() {
        return WinogradGemmDot2AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg,
          problemK,
          s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize) ? s.dev->info.subgroupSize : 0u);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmDot2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.dot2AccF16BM,
      0,
      VulkanKernels::DOT2_BK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }

  KernelBench Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int paddedSpatialSize,
    int convSize,
    int inChannels,
    int outChannels) {
    KernelBench result;
    if(
      iters < 1 || batchSize < 1 || nnXLen < 1 || nnYLen < 1 || inChannels < 1 || outChannels < 1 ||
      (convSize != 3 && convSize != 5) || s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return result;
    const bool is5x5 = convSize == 5;
    std::array<int32_t, 10> tile;
    convTileFromConfig(cfg, is5x5, kConv3x3Coopmat1TileFields, kConv5x5Coopmat1TileFields, tile);
    const int workgroupSize = tile[0], bm = tile[1], bn = tile[2], bk = tile[3], sgm = tile[4], sgn = tile[5],
              tm = tile[6], tn = tile[7], tk = tile[8], subgroupSize = tile[9];
    if(!coopmatShapeIsSupported(s.dev->info.coopmatShapes, tm, tn, tk))
      return result;
    if(!isConfigSupported(workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize))
      return result;
    const VkPhysicalDeviceLimits& limits = s.dev->info.properties.limits;
    if(
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupInvocations ||
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupSize[0] ||
      sharedBytes(bm, bn, bk) > limits.maxComputeSharedMemorySize || paddedSpatialSize % bm != 0 ||
      outChannels % bn != 0 || inChannels % 8 != 0)
      return result;

    try {
      const int flattenedK = convSize * convSize * inChannels;
      ComputeKernel kernel = build(
        s.device(), VK_NULL_HANDLE, cfg, convSize, flattenedK, false, inChannels % bk == 0, s.dev->info.subgroupSize);

      const size_t inputElts = (size_t)batchSize * paddedSpatialSize * inChannels;
      const size_t filterElts = (size_t)flattenedK * outChannels;
      const size_t outputElts = (size_t)batchSize * paddedSpatialSize * outChannels;
      std::vector<float> inputData(inputElts), filterData(filterElts);
      const uint32_t seed =
        (uint32_t)(nnXLen * 73856093u ^ nnYLen * 19349663u ^ inChannels * 83492791u ^ outChannels * 2654435761u);
      VulkanTuner::fillRandom(inputData, seed);
      VulkanTuner::fillRandom(filterData, seed ^ 0x9e3779b9u);
      std::vector<float> packedFilter =
        packStridedGemmBWeights(filterData, 1, outChannels, flattenedK, bn, bk, STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf filterBuf = s.makeInputBufFP(packedFilter);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);
      PC pc = {
        nnXLen,
        nnYLen,
        outChannels,
        outChannels,
        inChannels,
        paddedSpatialSize,
        0,
        outChannels * (paddedSpatialSize / 4),
        0};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(cctx, kernel, inputBuf.get(), filterBuf.get(), outputBuf.get(), convSize, pc, batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int paddedSpatialSize,
    int convSize,
    int inChannels,
    int outChannels) {
    KernelBench result;
    const bool is5x5 = convSize == 5;
    std::array<int32_t, 10> tile;
    convTileFromConfig(cfg, is5x5, kConv3x3Coopmat1AccF16TileFields, kConv5x5Coopmat1AccF16TileFields, tile);
    const int workgroupSize = tile[0], bm = tile[1], bn = tile[2], bk = tile[3], sgm = tile[4], sgn = tile[5],
              tm = tile[6], tn = tile[7], tk = tile[8], subgroupSize = tile[9];
    if(
      iters < 1 || batchSize < 1 || nnXLen < 1 || nnYLen < 1 || inChannels < 1 || outChannels < 1 ||
      (convSize != 3 && convSize != 5) || s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 ||
      !s.fp16Storage || !coopmatShapeIsSupported(s.dev->info.coopmatAccF16Shapes, tm, tn, tk) ||
      !isConfigSupported(workgroupSize, bm, bn, bk, sgm, sgn, tm, tn, tk, subgroupSize) ||
      paddedSpatialSize % bm != 0 || outChannels % bn != 0 || inChannels % 8 != 0)
      return result;
    const VkPhysicalDeviceLimits& limits = s.dev->info.properties.limits;
    if(
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupInvocations ||
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupSize[0] ||
      sharedBytes(bm, bn, bk) > limits.maxComputeSharedMemorySize)
      return result;
    try {
      const int flattenedK = convSize * convSize * inChannels;
      ComputeKernel kernel = build(
        s.device(), VK_NULL_HANDLE, cfg, convSize, flattenedK, false, inChannels % bk == 0, s.dev->info.subgroupSize);
      const size_t inputElts = (size_t)batchSize * paddedSpatialSize * inChannels;
      const size_t outputElts = (size_t)batchSize * paddedSpatialSize * outChannels;
      std::vector<float> inputData(inputElts), filterData((size_t)flattenedK * outChannels);
      const uint32_t seed =
        (uint32_t)(nnXLen * 73856093u ^ nnYLen * 19349663u ^ inChannels * 83492791u ^ outChannels * 2654435761u);
      VulkanTuner::fillRandom(inputData, seed);
      VulkanTuner::fillRandom(filterData, seed ^ 0x9e3779b9u);
      std::vector<float> packedFilter =
        packStridedGemmBWeights(filterData, 1, outChannels, flattenedK, bn, bk, STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
      VBuf inputBuf = s.makeInputBufFP(inputData), filterBuf = s.makeInputBufFP(packedFilter);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);
      PC pc = {
        nnXLen,
        nnYLen,
        outChannels,
        outChannels,
        inChannels,
        paddedSpatialSize,
        0,
        outChannels * (paddedSpatialSize / 4),
        0};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(cctx, kernel, inputBuf.get(), filterBuf.get(), outputBuf.get(), convSize, pc, batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };
      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger)
        s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmatAccF16 candidate failed: ") + e.what());
    }
    return result;
  }

  KernelBench Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int paddedSpatialSize,
    int convSize,
    int inChannels,
    int outChannels) {
    KernelBench result;
    const bool is5x5 = convSize == 5;
    std::array<int32_t, 4> tile;
    convTileFromConfig(cfg, is5x5, kConv3x3Coopmat2TileFields, kConv5x5Coopmat2TileFields, tile);
    const int workgroupSize = tile[0], bm = tile[1], bn = tile[2], bk = tile[3];
    if(
      iters < 1 || batchSize < 1 || s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage ||
      (convSize != 3 && convSize != 5) || inChannels % 8 != 0 || paddedSpatialSize % bm != 0 || outChannels % bn != 0 ||
      !isConfigSupported(workgroupSize, bm, bn, bk))
      return result;
    try {
      const int K = convSize * convSize * inChannels;
      const int packedK = K;
      ComputeKernel kernel = build(s.device(), VK_NULL_HANDLE, cfg, convSize, packedK, false);
      const size_t aElts = (size_t)batchSize * paddedSpatialSize * inChannels;
      const size_t cElts = (size_t)batchSize * paddedSpatialSize * outChannels;
      std::vector<float> a(aElts), b((size_t)K * outChannels);
      const uint32_t seed =
        (uint32_t)(nnXLen * 73856093u ^ nnYLen * 19349663u ^ inChannels * 83492791u ^ outChannels * 2654435761u);
      VulkanTuner::fillRandom(a, seed);
      VulkanTuner::fillRandom(b, seed ^ 0x9e3779b9u);
      b = packStridedGemmBWeights(b, 1, outChannels, packedK, bn, bk, STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
      VBuf aBuf = s.makeInputBufFP(a), bBuf = s.makeInputBufFP(b);
      VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, s.fp16Storage);
      PC pc = {
        nnXLen,
        nnYLen,
        outChannels,
        outChannels,
        inChannels,
        paddedSpatialSize,
        0,
        outChannels * (paddedSpatialSize / 4),
        0};
      auto record = [&]() {
        CmdCtx c{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(c, kernel, aBuf.get(), bBuf.get(), cBuf.get(), convSize, pc, batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
      };
      if(!timeAndCollect(s, iters, record, result, cBuf.get(), cElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger)
        s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat2 candidate failed: ") + e.what());
    }
    return result;
  }

  KernelBench Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int paddedSpatialSize,
    int convSize,
    int inChannels,
    int outChannels) {
    KernelBench result;
    const bool is5x5 = convSize == 5;
    std::array<int32_t, 4> tile;
    convTileFromConfig(cfg, is5x5, kConv3x3Coopmat2AccF16TileFields, kConv5x5Coopmat2AccF16TileFields, tile);
    const int workgroupSize = tile[0], bm = tile[1], bn = tile[2], bk = tile[3];
    if(
      iters < 1 || batchSize < 1 || s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage ||
      (convSize != 3 && convSize != 5) || inChannels % 8 != 0 || paddedSpatialSize % bm != 0 || outChannels % bn != 0 ||
      !isConfigSupported(workgroupSize, bm, bn, bk))
      return result;
    const VkPhysicalDeviceLimits& limits = s.dev->info.properties.limits;
    if(
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupInvocations ||
      (uint32_t)workgroupSize > limits.maxComputeWorkGroupSize[0] ||
      (size_t)s.dev->info.coopmat2ReservedSharedBytes + sharedBytes(bm, bk) > limits.maxComputeSharedMemorySize)
      return result;
    try {
      const int K = convSize * convSize * inChannels;
      ComputeKernel kernel = build(s.device(), VK_NULL_HANDLE, cfg, convSize, K, false);
      const size_t aElts = (size_t)batchSize * paddedSpatialSize * inChannels;
      const size_t cElts = (size_t)batchSize * paddedSpatialSize * outChannels;
      std::vector<float> a(aElts), b((size_t)K * outChannels);
      const uint32_t seed =
        (uint32_t)(nnXLen * 73856093u ^ nnYLen * 19349663u ^ inChannels * 83492791u ^ outChannels * 2654435761u);
      VulkanTuner::fillRandom(a, seed);
      VulkanTuner::fillRandom(b, seed ^ 0x9e3779b9u);
      b = packStridedGemmBWeights(b, 1, outChannels, K, bn, bk, STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
      VBuf aBuf = s.makeInputBufFP(a), bBuf = s.makeInputBufFP(b);
      VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, s.fp16Storage);
      PC pc = {
        nnXLen,
        nnYLen,
        outChannels,
        outChannels,
        inChannels,
        paddedSpatialSize,
        0,
        outChannels * (paddedSpatialSize / 4),
        0};
      auto record = [&]() {
        CmdCtx c{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(c, kernel, aBuf.get(), bBuf.get(), cBuf.get(), convSize, pc, batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
      };
      if(!timeAndCollect(s, iters, record, result, cBuf.get(), cElts, true, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger)
        s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat2AccF16 candidate failed: ") + e.what());
    }
    return result;
  }

  KernelBench WinogradGemmCoopmat1::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(s.dev->info.coopmatShapes, cfg.coopmat1TM, cfg.coopmat1TN, cfg.coopmat1TK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat1",
      [&]() {
        return WinogradGemmCoopmat1::build(s.device(), VK_NULL_HANDLE, cfg, problemK, s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmCoopmat1::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      true,
      cfg.coopmat1BM,
      cfg.coopmat1BN,
      cfg.coopmat1BK);
  }
  KernelBench WinogradGemmCoopmat1AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes, cfg.coopmat1AccF16TM, cfg.coopmat1AccF16TN, cfg.coopmat1AccF16TK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat1AccF16",
      [&]() {
        return WinogradGemmCoopmat1AccF16::build(s.device(), VK_NULL_HANDLE, cfg, problemK, s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) {
        WinogradGemmCoopmat1AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches);
      },
      true,
      cfg.coopmat1AccF16BM,
      cfg.coopmat1AccF16BN,
      cfg.coopmat1AccF16BK);
  }
  KernelBench GemmStridedCoopmat1Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatShapes, cfg.nhwcStridedCoopmat1TM, cfg.nhwcStridedCoopmat1TN, cfg.nhwcStridedCoopmat1TK))
      return KernelBench();
    if(!isConfigSupported(
         cfg.nhwcStridedCoopmat1WorkgroupSize,
         cfg.nhwcStridedCoopmat1BM,
         cfg.nhwcStridedCoopmat1BN,
         cfg.nhwcStridedCoopmat1BK,
         cfg.nhwcStridedCoopmat1SGM,
         cfg.nhwcStridedCoopmat1SGN,
         cfg.nhwcStridedCoopmat1TM,
         cfg.nhwcStridedCoopmat1TN,
         cfg.nhwcStridedCoopmat1TK,
         cfg.nhwcStridedCoopmat1SubgroupSize))
      return KernelBench();
    if(
      sharedBytes(
        cfg.nhwcStridedCoopmat1WorkgroupSize,
        cfg.nhwcStridedCoopmat1BM,
        cfg.nhwcStridedCoopmat1BN,
        cfg.nhwcStridedCoopmat1BK,
        cfg.nhwcStridedCoopmat1TM,
        cfg.nhwcStridedCoopmat1TN,
        cfg.nhwcStridedCoopmat1SubgroupSize) > s.dev->info.properties.limits.maxComputeSharedMemorySize)
      return KernelBench();
    const int aligned = (M % cfg.nhwcStridedCoopmat1BM == 0 && N % cfg.nhwcStridedCoopmat1BN == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedCoopmat1Nhwc",
      [&]() { return build(s.device(), VK_NULL_HANDLE, cfg, aligned, K, s.dev->info.subgroupSize, addToOutput); },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedPC& pc,
        int dispatchBatchSize) { dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize); },
      cfg.nhwcStridedCoopmat1BN,
      cfg.nhwcStridedCoopmat1BK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS,
      true);
  }

  KernelBench GemmStridedCoopmat1AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes,
         cfg.nhwcStridedCoopmat1AccF16TM,
         cfg.nhwcStridedCoopmat1AccF16TN,
         cfg.nhwcStridedCoopmat1AccF16TK))
      return KernelBench();
    if(!GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
         cfg.nhwcStridedCoopmat1AccF16WorkgroupSize,
         cfg.nhwcStridedCoopmat1AccF16BM,
         cfg.nhwcStridedCoopmat1AccF16BN,
         cfg.nhwcStridedCoopmat1AccF16BK,
         cfg.nhwcStridedCoopmat1AccF16SGM,
         cfg.nhwcStridedCoopmat1AccF16SGN,
         cfg.nhwcStridedCoopmat1AccF16TM,
         cfg.nhwcStridedCoopmat1AccF16TN,
         cfg.nhwcStridedCoopmat1AccF16TK,
         cfg.nhwcStridedCoopmat1AccF16SubgroupSize))
      return KernelBench();
    if(
      GemmStridedCoopmat1AccF16Nhwc::sharedBytes(
        cfg.nhwcStridedCoopmat1AccF16WorkgroupSize,
        cfg.nhwcStridedCoopmat1AccF16BM,
        cfg.nhwcStridedCoopmat1AccF16BN,
        cfg.nhwcStridedCoopmat1AccF16BK,
        cfg.nhwcStridedCoopmat1AccF16TM,
        cfg.nhwcStridedCoopmat1AccF16TN,
        cfg.nhwcStridedCoopmat1AccF16SubgroupSize) > s.dev->info.properties.limits.maxComputeSharedMemorySize)
      return KernelBench();
    const int32_t aligned =
      (M % cfg.nhwcStridedCoopmat1AccF16BM == 0 && N % cfg.nhwcStridedCoopmat1AccF16BN == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedCoopmat1AccF16Nhwc",
      [&]() {
        return GemmStridedCoopmat1AccF16Nhwc::build(
          s.device(), VK_NULL_HANDLE, cfg, aligned, K, s.dev->info.subgroupSize, addToOutput);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedPC& pc,
        int dispatchBatchSize) {
        GemmStridedCoopmat1AccF16Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedCoopmat1AccF16BN,
      cfg.nhwcStridedCoopmat1AccF16BK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS,
      true);
  }

  KernelBench GemmStridedCoopmat2Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(
      s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage ||
      !isConfigSupported(
        cfg.nhwcStridedCoopmat2WorkgroupSize,
        cfg.nhwcStridedCoopmat2BM,
        cfg.nhwcStridedCoopmat2BN,
        cfg.nhwcStridedCoopmat2BK))
      return KernelBench();
    const int aligned = (M % cfg.nhwcStridedCoopmat2BM == 0 && N % cfg.nhwcStridedCoopmat2BN == 0) ? 1 : 0;
    const int kAligned = K % cfg.nhwcStridedCoopmat2BK == 0 ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedCoopmat2Nhwc",
      [&]() { return build(s.device(), VK_NULL_HANDLE, cfg, aligned, kAligned, K, addToOutput); },
      [&](
        const CmdCtx& c,
        const ComputeKernel& kernel,
        VulkanBuffer* a,
        VulkanBuffer* b,
        VulkanBuffer* out,
        const GemmStridedPC& pc,
        int batches) { dispatch(c, kernel, a, b, out, K, pc, batches); },
      cfg.nhwcStridedCoopmat2BN,
      cfg.nhwcStridedCoopmat2BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS,
      true);
  }

  KernelBench GemmStridedCoopmat2AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(
      s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage ||
      !isConfigSupported(
        cfg.nhwcStridedCoopmat2AccF16WorkgroupSize,
        cfg.nhwcStridedCoopmat2AccF16BM,
        cfg.nhwcStridedCoopmat2AccF16BN,
        cfg.nhwcStridedCoopmat2AccF16BK))
      return KernelBench();
    const int aligned = (M % cfg.nhwcStridedCoopmat2AccF16BM == 0 && N % cfg.nhwcStridedCoopmat2AccF16BN == 0) ? 1 : 0;
    const int kAligned = K % cfg.nhwcStridedCoopmat2AccF16BK == 0 ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedCoopmat2AccF16Nhwc",
      [&]() { return build(s.device(), VK_NULL_HANDLE, cfg, aligned, kAligned, K, addToOutput); },
      [&](
        const CmdCtx& c,
        const ComputeKernel& kernel,
        VulkanBuffer* a,
        VulkanBuffer* b,
        VulkanBuffer* out,
        const GemmStridedPC& pc,
        int batches) { dispatch(c, kernel, a, b, out, K, pc, batches); },
      cfg.nhwcStridedCoopmat2AccF16BN,
      cfg.nhwcStridedCoopmat2AccF16BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS,
      true);
  }

  // NHWC K-inner A and N-contiguous C use the transpose-based correctness oracle.
  KernelBench GemmStridedDot2Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16 || !s.fp16Storage)
      return KernelBench();
    if(!GemmStridedDot2Nhwc::isConfigSupported(
         cfg.nhwcStridedDot2WorkgroupSize,
         cfg.nhwcStridedDot2BM,
         cfg.nhwcStridedDot2BN,
         cfg.nhwcStridedDot2SGM,
         cfg.nhwcStridedDot2SGN,
         cfg.nhwcStridedDot2SGMIter,
         cfg.nhwcStridedDot2TM,
         cfg.nhwcStridedDot2TN,
         cfg.nhwcStridedDot2SubgroupSize))
      return KernelBench();
    // DOT2's BK is a fixed shader #define (VulkanKernels::DOT2_BK), not a tuned
    // field, so alignment reads the NHWC BM/BN fields directly.
    const int32_t aligned = (M % cfg.nhwcStridedDot2BM == 0 && N % cfg.nhwcStridedDot2BN == 0) ? 1 : 0;
    const int32_t kAligned = (K % VulkanKernels::DOT2_BK == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedDot2Nhwc",
      [&]() {
        return GemmStridedDot2Nhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg,
          aligned,
          VulkanKernels::DOT2_PACKED_B,
          kAligned,
          K,
          addToOutput,
          s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize) ? s.dev->info.subgroupSize : 0u);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedPC& pc,
        int dispatchBatchSize) {
        GemmStridedDot2Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedDot2BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
  }

  // Native-NHWC companion to GemmStridedDot2AccF16::bench.
  KernelBench GemmStridedDot2AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!GemmStridedDot2AccF16Nhwc::isConfigSupported(
         cfg.nhwcStridedDot2AccF16WorkgroupSize,
         cfg.nhwcStridedDot2AccF16BM,
         cfg.nhwcStridedDot2AccF16BN,
         cfg.nhwcStridedDot2AccF16SGM,
         cfg.nhwcStridedDot2AccF16SGN,
         cfg.nhwcStridedDot2AccF16SGMIter,
         cfg.nhwcStridedDot2AccF16TM,
         cfg.nhwcStridedDot2AccF16TN,
         cfg.nhwcStridedDot2AccF16SubgroupSize))
      return KernelBench();
    const int32_t aligned = (M % cfg.nhwcStridedDot2AccF16BM == 0 && N % cfg.nhwcStridedDot2AccF16BN == 0) ? 1 : 0;
    const int32_t kAligned = (K % VulkanKernels::DOT2_BK == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedDot2AccF16Nhwc",
      [&]() {
        return GemmStridedDot2AccF16Nhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg,
          aligned,
          VulkanKernels::DOT2_PACKED_B,
          kAligned,
          K,
          addToOutput,
          s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize) ? s.dev->info.subgroupSize : 0u);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedPC& pc,
        int dispatchBatchSize) {
        GemmStridedDot2AccF16Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedDot2AccF16BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
  }

  KernelBench WinogradGemmCoopmat2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileSupported(
         s.dev->info,
         s.dev->info.coopmat2FlexShapes,
         Coopmat2Tile{cfg.coopmat2WorkgroupSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK}))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat2",
      [&]() { return WinogradGemmCoopmat2::build(s.device(), VK_NULL_HANDLE, cfg, problemK); },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmCoopmat2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      true,
      cfg.coopmat2BM,
      cfg.coopmat2BN,
      cfg.coopmat2BK,
      WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
  }
  KernelBench WinogradGemmCoopmat2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileSupported(
         s.dev->info,
         s.dev->info.coopmat2AccF16FlexShapes,
         Coopmat2Tile{cfg.coopmat2AccF16WorkgroupSize, cfg.coopmat2AccF16BM, cfg.coopmat2AccF16BN, cfg.coopmat2AccF16BK}))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat2AccF16",
      [&]() { return WinogradGemmCoopmat2AccF16::build(s.device(), VK_NULL_HANDLE, cfg, problemK); },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) {
        WinogradGemmCoopmat2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches);
      },
      true,
      cfg.coopmat2AccF16BM,
      cfg.coopmat2AccF16BN,
      cfg.coopmat2AccF16BK,
      WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
  }
  KernelBench GemmDirectFP32::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = GemmDirectFP32::build(s.device(), VK_NULL_HANDLE, cfg, (uint32_t)problemK);
      size_t aElts = (size_t)problemM * problemK;
      size_t bElts = (size_t)problemN * problemK;
      size_t cElts = (size_t)problemM * problemN;
      std::vector<float> aData(aElts), bData(bElts);
      uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u ^ 12345u);
      VulkanTuner::fillRandom(aData, seed);
      VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);
      VBuf aBuf = s.makeInputBufFP(aData);
      VBuf bBuf = s.makeInputBufFP(bData);
      VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, false);  // GemmDirectFP32 output is always FP32
      GemmDirectFP32::PC pc = {problemM, problemN};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        GemmDirectFP32::dispatch(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), problemK, pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
      };
      if(!timeAndCollect(s, iters, recordOne, result, cBuf.get(), cElts, false, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: gemmDirect candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench GPoolReductionNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = GPoolReductionNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t outputElts = (size_t)problemBatchSize * problemChannels * 3;  // mean, mean*sqrt, max
      size_t maskElts = (size_t)problemBatchSize * problemXySize;
      size_t maskSumElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskData(maskElts, 1.0f), maskSumData(maskSumElts);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 73856093u ^ problemXySize * 19349663u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, false);  // gpool output is always FP32
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);

      GPoolReductionNhwc::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        GPoolReductionNhwc::dispatch(
          cctx, kernel, inputBuf.get(), outputBuf.get(), maskBuf.get(), maskSumBuf.get(), pc, problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, false, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: gpoolReductionNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench ValueHeadPoolNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = ValueHeadPoolNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t outputElts = (size_t)problemBatchSize * problemChannels * 3;  // mean, mean*t*0.1, mean*(t^2*0.01-0.1)
      size_t maskSumElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskSumData(maskSumElts);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 47093279u ^ problemXySize * 30000023u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, false);  // output is always FP32
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);

      ValueHeadPoolNhwc::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        ValueHeadPoolNhwc::dispatch(
          cctx, kernel, inputBuf.get(), outputBuf.get(), maskSumBuf.get(), pc, problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      if(!timeAndCollect(s, iters, recordOne, result, outputBuf.get(), outputElts, false, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: valueHeadPoolNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench SpatialRMSNormNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int tile = cfg.spatialRMSNormNhwcTile;
      const int spatialSize = problemChannels * problemXySize;
      const int numSpatialWorkgroups = (spatialSize + tile - 1) / tile;

      const bool canRequireSize =
        s.dev != nullptr && s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize);
      const bool useSpatialRMSNormPass2Subgroup = s.dev != nullptr && canUseRMSNormSubgroupVariant(
                                                                        s.dev->info.supportsSubgroupShuffleCompute,
                                                                        canRequireSize,
                                                                        s.dev->info.subgroupSize,
                                                                        (uint32_t)tile,
                                                                        /*wgChannelGroup=*/1u);
      const uint32_t rmsNormPass2SubgroupSize = useSpatialRMSNormPass2Subgroup ? s.dev->info.subgroupSize : 0u;
      std::array<ComputeKernel, 3> kernels = SpatialRMSNormNhwc::build(
        s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg, useSpatialRMSNormPass2Subgroup, rmsNormPass2SubgroupSize);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t maskElts = (size_t)problemBatchSize * problemXySize;
      size_t maskSumElts = (size_t)problemBatchSize;
      size_t gammaBetaElts = (size_t)problemChannels;
      size_t partialElts = (size_t)problemBatchSize * numSpatialWorkgroups;
      size_t scalarElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskData(maskElts, 1.0f), maskSumData(maskSumElts);
      std::vector<float> gammaData(gammaBetaElts, 1.0f), betaData(gammaBetaElts, 0.0f);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 11400714u ^ problemXySize * 2654435u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), inputElts, s.fp16Storage);
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);
      VBuf gammaBuf = s.makeInputBuf(gammaData);
      VBuf betaBuf = s.makeInputBuf(betaData);
      VBuf partialsBuf = makeDeviceBuf(s.device(), s.memProps(), partialElts, false);
      VBuf scalarBuf = makeDeviceBuf(s.device(), s.memProps(), scalarElts, false);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        SpatialRMSNormNhwc::PC pc = {problemChannels, problemXySize, 1e-5f};
        SpatialRMSNormNhwc::dispatch(
          cctx,
          kernels,
          inputBuf.get(),
          outputBuf.get(),
          gammaBuf.get(),
          betaBuf.get(),
          maskBuf.get(),
          maskSumBuf.get(),
          partialsBuf.get(),
          scalarBuf.get(),
          pc,
          problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        for(auto& k: kernels)
          k.destroy(s.device());
        return result;
      }
      result.output.resize(inputElts);
      s.downloadFloatsFP(outputBuf.get(), result.output, inputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      for(auto& k: kernels)
        k.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: spatialRMSNorm candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench
  AttentionCoopmat1Bench::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int32_t kvChunkCount, int iters)
    const {
    KernelBench result;
    if(iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute || !s.dev->info.supportsCoopmat1F16)
      return result;
    if(
      !coopmatShapeIsSupported(
        s.dev->info.coopmatShapes, cfg.attnNhwcCoopmat1TM, cfg.attnNhwcCoopmat1TN, cfg.attnNhwcCoopmat1TK) ||
      cfg.attnNhwcCoopmat1SubgroupSize != (int32_t)s.dev->info.subgroupSize ||
      !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      const AttentionBenchProblem p{
        problemBatchSize,
        problemNumTokens,
        problemValidTokens,
        problemHeadDim,
        problemVHeadDim,
        problemNumHeads,
        problemNumKVHeads,
        problemUseRope,
        problemLearnableRope};
      ComputeKernel kernel = AttentionCoopmat1Nhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        cfg,
        p.numTokens,
        p.headDim,
        p.vHeadDim,
        p.useRope,
        p.learnableRope,
        s.dev->info.subgroupSize,
        kvChunkCount,
        usesMaintenance1());
      AttentionBenchBuffers b = makeAttentionBenchBuffers(s, p, false, true);

      const int numBH = p.batchSize * p.numHeads;
      VBuf partialsBuf;
      VBuf statsBuf;
      ComputeKernel resolveKernel{};
      if(kvChunkCount > 1) {
        size_t partialsElts =
          AttentionCoopmat1Nhwc::partialsBytes(p.numTokens, numBH, kvChunkCount, p.vHeadDim) / sizeof(float);
        size_t statsElts = AttentionCoopmat1Nhwc::statsBytes(p.numTokens, numBH, kvChunkCount) / sizeof(float);
        partialsBuf = makeDeviceBuf(s.device(), s.memProps(), partialsElts, false);
        statsBuf = makeDeviceBuf(s.device(), s.memProps(), statsElts, false);
        resolveKernel = AttentionSplitKResolveNhwc::build(s.device(), VK_NULL_HANDLE, p.vHeadDim);
      }
      AttentionTiled::PC pc = {
        p.numHeads,
        p.numKVHeads,
        1.0f / sqrtf((float)p.headDim),
        numBH,
      };
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionCoopmat1Nhwc::dispatch(
          cctx,
          kernel,
          b.q.get(),
          b.k.get(),
          b.v.get(),
          b.out.get(),
          b.mask.get(),
          p.useRope ? b.cos.get() : nullptr,
          p.useRope ? b.sin.get() : nullptr,
          p.numTokens,
          pc,
          kvChunkCount > 1 ? partialsBuf.get() : nullptr,
          kvChunkCount > 1 ? statsBuf.get() : nullptr);
        if(kvChunkCount > 1) {
          VulkanHelpers::cmdComputeBarrier(s.cmd, partialsBuf->buffer);
          VulkanHelpers::cmdComputeBarrier(s.cmd, statsBuf->buffer);
          AttentionSplitKResolveNhwc::PC resolvePC = {p.numHeads, p.numTokens, kvChunkCount};
          AttentionSplitKResolveNhwc::dispatch(
            cctx, resolveKernel, partialsBuf.get(), statsBuf.get(), b.out.get(), p.numTokens, numBH, resolvePC);
        }
        VulkanHelpers::cmdComputeBarrier(s.cmd, b.out->buffer);
      };
      if(!timeAttentionBench(s, iters, recordOne, b, p, result, kernel)) {
        if(kvChunkCount > 1)
          resolveKernel.destroy(s.device());
        return result;
      }
      if(kvChunkCount > 1)
        resolveKernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnCoopmat1Nhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionCoopmat2AccF32Bench::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    KernelBench result;
    if(
      iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute || !s.dev->info.supportsCoopmat2Attention ||
      !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      const AttentionBenchProblem p{
        problemBatchSize,
        problemNumTokens,
        problemValidTokens,
        problemHeadDim,
        problemVHeadDim,
        problemNumHeads,
        problemNumKVHeads,
        problemUseRope,
        problemLearnableRope};
      ComputeKernel kernel = AttentionCoopmat2AccF32Nhwc::build(
        s.device(), VK_NULL_HANDLE, cfg, p.numTokens, p.headDim, p.vHeadDim, p.useRope, p.learnableRope);
      AttentionBenchBuffers b = makeAttentionBenchBuffers(s, p, true, true);
      AttentionTiled::PC pc = {p.numHeads, p.numKVHeads, 1.0f / sqrtf((float)p.headDim), p.batchSize * p.numHeads};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionCoopmat2AccF32Nhwc::dispatch(
          cctx,
          kernel,
          b.q.get(),
          b.k.get(),
          b.v.get(),
          b.out.get(),
          b.mask.get(),
          p.useRope ? b.cos.get() : nullptr,
          p.useRope ? b.sin.get() : nullptr,
          p.numTokens,
          pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, b.out->buffer);
      };
      if(!timeAttentionBench(s, iters, recordOne, b, p, result, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnCoopmat2Nhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionDot2AccF32Bench::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    KernelBench result;
    if(
      iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute || !s.dev->info.supportsDot2F16 ||
      !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      const AttentionBenchProblem p{
        problemBatchSize,
        problemNumTokens,
        problemValidTokens,
        problemHeadDim,
        problemVHeadDim,
        problemNumHeads,
        problemNumKVHeads,
        problemUseRope,
        problemLearnableRope};
      ComputeKernel kernel = AttentionDot2AccF32Nhwc::build(
        s.device(), VK_NULL_HANDLE, cfg, p.numTokens, p.headDim, p.vHeadDim, p.useRope, p.learnableRope);
      AttentionBenchBuffers b = makeAttentionBenchBuffers(s, p, false, true);
      AttentionTiled::PC pc = {p.numHeads, p.numKVHeads, 1.0f / sqrtf((float)p.headDim), p.batchSize * p.numHeads};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionDot2AccF32Nhwc::dispatch(
          cctx,
          kernel,
          b.q.get(),
          b.k.get(),
          b.v.get(),
          b.out.get(),
          b.mask.get(),
          p.useRope ? b.cos.get() : nullptr,
          p.useRope ? b.sin.get() : nullptr,
          p.numTokens,
          pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, b.out->buffer);
      };
      if(!timeAttentionBench(s, iters, recordOne, b, p, result, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnDot2AccF32Nhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionTiled::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const AttentionBenchProblem p{
        problemBatchSize,
        problemNumTokens,
        problemValidTokens,
        problemHeadDim,
        problemVHeadDim,
        problemNumHeads,
        problemNumKVHeads,
        problemUseRope,
        problemLearnableRope};
      ComputeKernel kernel = AttentionTiledNhwc::build(
        s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg, p.numTokens, p.headDim, p.vHeadDim, p.useRope, p.learnableRope);
      // mask is bound at both the FP32 and FP16 slots via bindingMap, so its
      // storage width must match s.fp16Storage — use the FP-aware upload.
      AttentionBenchBuffers b = makeAttentionBenchBuffers(s, p, false, s.fp16Storage);
      AttentionTiled::PC pc = {p.numHeads, p.numKVHeads, 1.0f / sqrtf((float)p.headDim), p.batchSize * p.numHeads};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionTiledNhwc::dispatch(
          cctx,
          kernel,
          b.q.get(),
          b.k.get(),
          b.v.get(),
          b.out.get(),
          b.mask.get(),
          p.useRope ? b.cos.get() : nullptr,
          p.useRope ? b.sin.get() : nullptr,
          p.numTokens,
          pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, b.out->buffer);
      };
      if(!timeAttentionBench(s, iters, recordOne, b, p, result, kernel))
        return result;
      return result;
    } catch(const std::exception& e) {
      // A lost device is not a bad candidate: it invalidates every later
      // bench too, so stop the sweep instead of logging thousands of
      // indistinguishable "candidate failed" lines.
      if(dynamic_cast<const VulkanHelpers::DeviceLostError*>(&e) != nullptr)
        throw;
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnTiled candidate failed: ") + e.what());
      return result;
    }
  }

}  // namespace VulkanKernels
#endif  // USE_VULKAN_BACKEND

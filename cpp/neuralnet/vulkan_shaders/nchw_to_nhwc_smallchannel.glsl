// Direct small-channel repack of a spatial tensor from NCHW [N,C,XY] to
// NHWC [N,XY,C]. C=2 and C=4 use packed vec4 stores; other small channel
// counts use one scalar load/store per invocation.
#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(constant_id = 0) const int USE_FP16_STORAGE = 0;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int inputChannels;
  int outputChannels;
  int paddedSpatialSize;
  int batchSize;
}
pc;

layout(binding = 0) readonly buffer InputBuf {
  float data[];
}
inputBuf;
layout(binding = 1) writeonly buffer OutputBuf {
  float data[];
}
outputBuf;
layout(binding = 2) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 3) writeonly buffer OutputBufH {
  float16_t data[];
}
outputBufH;
layout(binding = 4) writeonly buffer OutputBufV4 {
  vec4 data[];
}
outputBufV4;
layout(binding = 5) writeonly buffer OutputBufHV4 {
  f16vec4 data[];
}
outputBufHV4;

float loadInput(uint idx) {
  return USE_FP16_STORAGE == 1 ? float(inputBufH.data[idx]) : inputBuf.data[idx];
}

void storeOutput(uint idx, float value) {
  if(USE_FP16_STORAGE == 1)
    outputBufH.data[idx] = float16_t(value);
  else
    outputBuf.data[idx] = value;
}

void storeOutput4(uint idx, vec4 value) {
  if(USE_FP16_STORAGE == 1)
    outputBufHV4.data[idx] = f16vec4(value);
  else
    outputBufV4.data[idx] = value;
}

void main() {
  const uint inputChannels = uint(pc.inputChannels);
  const uint outputChannels = uint(pc.outputChannels);
  const uint spatialSize = uint(pc.paddedSpatialSize);
  const bool packed = inputChannels == outputChannels && (outputChannels == 2u || outputChannels == 4u);

  if(packed && outputChannels == 4u) {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = uint(pc.batchSize) * spatialSize;
    if(idx >= total)
      return;
    const uint xy = idx % spatialSize;
    const uint n = idx / spatialSize;
    storeOutput4(
      idx,
      vec4(
        loadInput((n * 4u + 0u) * spatialSize + xy),
        loadInput((n * 4u + 1u) * spatialSize + xy),
        loadInput((n * 4u + 2u) * spatialSize + xy),
        loadInput((n * 4u + 3u) * spatialSize + xy)));
    return;
  }

  if(packed && outputChannels == 2u) {
    const uint pairIdx = gl_GlobalInvocationID.x;
    const uint pairsPerBatch = spatialSize / 2u;
    const uint totalPairs = uint(pc.batchSize) * pairsPerBatch;
    if(pairIdx >= totalPairs)
      return;
    const uint n = pairIdx / pairsPerBatch;
    const uint xy = (pairIdx % pairsPerBatch) * 2u;
    storeOutput4(
      pairIdx,
      vec4(
        loadInput((n * 2u + 0u) * spatialSize + xy),
        loadInput((n * 2u + 1u) * spatialSize + xy),
        loadInput((n * 2u + 0u) * spatialSize + xy + 1u),
        loadInput((n * 2u + 1u) * spatialSize + xy + 1u)));
    return;
  }

  const uint idx = gl_GlobalInvocationID.x;
  const uint total = uint(pc.batchSize) * outputChannels * spatialSize;
  if(idx >= total)
    return;
  const uint xy = idx % spatialSize;
  const uint tmp = idx / spatialSize;
  const uint c = tmp % outputChannels;
  const uint n = tmp / outputChannels;
  storeOutput(
    (n * spatialSize + xy) * outputChannels + c,
    c < inputChannels ? loadInput((n * inputChannels + c) * spatialSize + xy) : 0.0);
}

// GLSL Compute Shader: Spatial RMSNorm Pass 3 - apply normalization with gamma/beta/mask
// Ports OpenCLKernels::transformerSpatialRMSNormApply
// Dispatch: global(APPLY_ELTS_PER_THREAD * localSize * numGroups, batchSize, 1)

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int APPLY_ELTS_PER_THREAD = 4;
layout(constant_id = 1) const int USE_FP16_STORAGE = 0;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
  float epsilon;
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
layout(binding = 2) readonly buffer GammaBuf {
  float data[];
}
gammaBuf;
layout(binding = 3) readonly buffer BetaBuf {
  float data[];
}
betaBuf;
layout(binding = 4) readonly buffer MaskBuf {
  float data[];
}
maskBuf;
layout(binding = 5) readonly buffer MaskSumBuf {
  float data[];
}
maskSumBuf;
layout(binding = 6) readonly buffer SumSqBuf {
  float data[];
}
sumSqBuf;
layout(binding = 7) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 8) writeonly buffer OutBufH {
  float16_t data[];
}
outBufH;
layout(binding = 9) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

float ldIn(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}
float ldMask(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}
void stOut(int idx, float v) {
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = float16_t(v);
  else
    outputBuf.data[idx] = v;
}

void main() {
  int n = int(gl_GlobalInvocationID.y);

  float mSum = maskSumBuf.data[n];
  float denom = mSum * float(pc.numChannels);
  float rms = inversesqrt(sumSqBuf.data[n] / denom + pc.epsilon);

  int totalElems = pc.numChannels * pc.paddedSpatialSize;
  int baseIdx = int(gl_WorkGroupID.x) * int(gl_WorkGroupSize.x) * APPLY_ELTS_PER_THREAD + int(gl_LocalInvocationID.x);

  for(int e = 0; e < APPLY_ELTS_PER_THREAD; e++) {
    int idx = baseIdx + int(gl_WorkGroupSize.x) * e;
    if(idx < totalElems) {
      int c = idx / pc.paddedSpatialSize;
      int xy = idx % pc.paddedSpatialSize;
      float maskVal = ldMask(n * pc.paddedSpatialSize + xy);
      float val = ldIn((n * pc.numChannels + c) * pc.paddedSpatialSize + xy);
      float result = (val * rms * gammaBuf.data[c] + betaBuf.data[c]) * maskVal;
      stOut((n * pc.numChannels + c) * pc.paddedSpatialSize + xy, result);
    }
  }
}

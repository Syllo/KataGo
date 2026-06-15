// GLSL Compute Shader: Value head global pool
// Ports OpenCLKernels::valueHeadPoolChannelsNCHW
// output[n, c, 0] = mean
// output[n, c, 1] = mean * (sqrt(maskSum[n]) - 14) * 0.1
// output[n, c, 2] = mean * ((sqrt(maskSum[n]) - 14)^2 * 0.01 - 0.1)
// Dispatch: global(XYSTRIDE, numChannels, batchSize)

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int XYSTRIDE = 64;
layout(constant_id = 1) const int USE_FP16_STORAGE = 0;

layout(local_size_x_id = 0) in;
layout(local_size_y = 1) in;
layout(local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
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
layout(binding = 2) readonly buffer MaskSumBuf {
  float data[];
}
maskSumBuf;
layout(binding = 3) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;

shared float partialSums[XYSTRIDE];

float ldIn(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}

void main() {
  int xyBase = int(gl_LocalInvocationID.x);
  int c = int(gl_GlobalInvocationID.y);
  int n = int(gl_GlobalInvocationID.z);

  float sum = 0.0;
  if(c < pc.numChannels) {
    for(int xy = xyBase; xy < pc.paddedSpatialSize; xy += XYSTRIDE) {
      sum += ldIn((n * pc.numChannels + c) * pc.paddedSpatialSize + xy);
    }
  }
  partialSums[xyBase] = sum;

  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
    barrier();
    if(xyBase < span)
      partialSums[xyBase] += partialSums[xyBase + span];
  }
  barrier();

  if(c < pc.numChannels && xyBase == 0) {
    float finalSum = partialSums[0];
    float div = maskSumBuf.data[n];
    float sqrtDiv = sqrt(div);
    float mean = finalSum / div;
    float t = sqrtDiv - 14.0;

    int outBase = n * pc.numChannels * 3 + c;
    outputBuf.data[outBase] = mean;
    outputBuf.data[outBase + pc.numChannels] = mean * t * 0.1;
    outputBuf.data[outBase + pc.numChannels * 2] = mean * (t * t * 0.01 - 0.1);
  }
}

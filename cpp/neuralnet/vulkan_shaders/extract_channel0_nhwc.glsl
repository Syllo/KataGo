// GLSL Compute Shader: Extract channel 0 from an NHWC tensor to a scalar mask.
// mask[n, xy] = input[n, xy, 0]. The mask has no channel-layout distinction.
// Dispatch: global(paddedSpatialSize, batchSize, 1)

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int USE_FP16_STORAGE = 0;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
}
pc;

layout(binding = 0) readonly buffer InputBuf {
  float data[];
}
inputBuf;
layout(binding = 1) writeonly buffer MaskBuf {
  float data[];
}
maskBuf;
layout(binding = 2) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 3) writeonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

void main() {
  int xy = int(gl_GlobalInvocationID.x);
  int n = int(gl_GlobalInvocationID.y);
  if(xy >= pc.paddedSpatialSize)
    return;

  int inputIdx = (n * pc.paddedSpatialSize + xy) * pc.numChannels;
  int maskIdx = n * pc.paddedSpatialSize + xy;
  if(USE_FP16_STORAGE == 1)
    maskBufH.data[maskIdx] = inputBufH.data[inputIdx];
  else
    maskBuf.data[maskIdx] = inputBuf.data[inputIdx];
}

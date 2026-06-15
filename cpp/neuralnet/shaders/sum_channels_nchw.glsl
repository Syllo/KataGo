// GLSL Compute Shader: Sum spatial positions per channel → maskSum
// Ports OpenCLKernels::sumChannelsNCHW
// maskSum[n] = sum over xy of mask[n, xy]  (spatial sum, not channel sum)
// Dispatch: global(1, batchSize, 1) — one workgroup per n, reduced across local x lanes

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
  int paddedSpatialSize;
  int batchSize;
}
pc;

layout(binding = 0) readonly buffer MaskBuf {
  float data[];
}
maskBuf;
layout(binding = 1) writeonly buffer MaskSumBuf {
  float data[];
}
maskSumBuf;
layout(binding = 2) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

shared float partials[64];

float ldMask(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}

void main() {
  int lid = int(gl_LocalInvocationID.x);
  int n = int(gl_WorkGroupID.y);
  if(n >= pc.batchSize)
    return;

  int base = n * pc.paddedSpatialSize;
  float sum = 0.0;
  for(int xy = lid; xy < pc.paddedSpatialSize; xy += 64) {
    sum += ldMask(base + xy);
  }

  partials[lid] = sum;
  barrier();
  for(int span = 32; span > 0; span >>= 1) {
    if(lid < span) {
      sum += partials[lid + span];
      partials[lid] = sum;
    }
    barrier();
  }
  if(lid == 0)
    maskSumBuf.data[n] = sum;
}

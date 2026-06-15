// GLSL Compute Shader: Spatial RMSNorm pass 1, NHWC layout.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int TILE_SIZE = 64;
layout(constant_id = 1) const int USE_FP16_STORAGE = 0;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
  int tilesPerGroup;
  int numCHWWorkgroups;
} pc;

layout(binding = 0) readonly buffer InputBuf { float data[]; } inputBuf;
layout(binding = 1) readonly buffer MaskBuf { float data[]; } maskBuf;
layout(binding = 2) writeonly buffer PartialsBuf { float data[]; } partialsBuf;
layout(binding = 3) readonly buffer InputBufH { float16_t data[]; } inputBufH;
layout(binding = 4) readonly buffer MaskBufH { float16_t data[]; } maskBufH;
shared float partials[TILE_SIZE];

float ldIn(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}
float ldMask(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}

void main() {
  int lid = int(gl_LocalInvocationID.x);
  int groupIdx = int(gl_WorkGroupID.x);
  int n = int(gl_GlobalInvocationID.y);
  int totalElems = pc.numChannels * pc.paddedSpatialSize;
  int chunkStart = groupIdx * TILE_SIZE * pc.tilesPerGroup;
  float acc = 0.0;
  for(int t = 0; t < pc.tilesPerGroup; t++) {
    int idx = chunkStart + t * TILE_SIZE + lid;
    if(idx < totalElems) {
      int xy = idx / pc.numChannels;
      float val = ldIn(n * totalElems + idx) * ldMask(n * pc.paddedSpatialSize + xy);
      acc += val * val;
    }
  }
  partials[lid] = acc;
  for(int span = TILE_SIZE / 2; span > 0; span /= 2) {
    barrier();
    if(lid < span)
      partials[lid] += partials[lid + span];
  }
  if(lid == 0)
    partialsBuf.data[n * pc.numCHWWorkgroups + groupIdx] = partials[0];
}

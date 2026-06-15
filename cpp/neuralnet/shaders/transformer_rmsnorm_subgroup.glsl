// GLSL Compute Shader: Per-position RMSNorm for transformer layers (subgroup-shuffle variant)
// Ports OpenCLKernels::transformerRMSNorm
// input/output: [N, C, paddedSpatialSize] NCHW
// Dispatch: global(WG_XY_SIZE * numGroups, batchSize, 1)
// Each workgroup handles WG_XY_SIZE xy positions

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int WG_XY_SIZE = 8;
layout(constant_id = 1) const int WG_C_SIZE = 8;
layout(constant_id = 2) const int C_PER_THREAD = 1;
layout(constant_id = 3) const int USE_FP16_STORAGE = 0;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;  // WG_XY_SIZE * WG_C_SIZE

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
layout(binding = 2) readonly buffer WeightBuf {
  float data[];
}
weightBuf;
layout(binding = 3) readonly buffer BetaBuf {
  float data[];
}
betaBuf;
layout(binding = 4) readonly buffer MaskBuf {
  float data[];
}
maskBuf;
layout(binding = 5) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 6) writeonly buffer OutBufH {
  float16_t data[];
}
outBufH;
layout(binding = 7) readonly buffer MaskBufH {
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
  int lid = int(gl_LocalInvocationID.x);
  int lid_xy = lid / WG_C_SIZE;
  int lid_c = lid % WG_C_SIZE;
  int xy = int(gl_WorkGroupID.x) * WG_XY_SIZE + lid_xy;
  int n = int(gl_GlobalInvocationID.y);

  float maskVal = (xy < pc.paddedSpatialSize) ? ldMask(n * pc.paddedSpatialSize + xy) : 0.0;

  float acc = 0.0;
  for(int base = lid_c * C_PER_THREAD; base < pc.numChannels; base += WG_C_SIZE * C_PER_THREAD) {
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < pc.numChannels && xy < pc.paddedSpatialSize) {
        float val = ldIn((n * pc.numChannels + c) * pc.paddedSpatialSize + xy) * maskVal;
        acc += val * val;
      }
    }
  }

  for(int span = WG_C_SIZE / 2; span > 0; span >>= 1) {
    acc += subgroupShuffleXor(acc, span);
  }
  int subgroupLane = int(gl_SubgroupInvocationID);
  int groupBase = (subgroupLane / WG_C_SIZE) * WG_C_SIZE;
  float reduced = subgroupShuffle(acc, groupBase);
  float rms = inversesqrt(reduced / float(pc.numChannels) + pc.epsilon);

  for(int base = lid_c * C_PER_THREAD; base < pc.numChannels; base += WG_C_SIZE * C_PER_THREAD) {
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < pc.numChannels && xy < pc.paddedSpatialSize) {
        float val = ldIn((n * pc.numChannels + c) * pc.paddedSpatialSize + xy);
        float result = (val * rms * weightBuf.data[c] + betaBuf.data[c]) * maskVal;
        stOut((n * pc.numChannels + c) * pc.paddedSpatialSize + xy, result);
      }
    }
  }
}

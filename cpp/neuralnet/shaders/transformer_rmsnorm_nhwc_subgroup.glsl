// GLSL Compute Shader: Per-position RMSNorm, NHWC subgroup-shuffle variant.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

// Match a full 32-lane subgroup to one contiguous NHWC channel run.
layout(constant_id = 0) const int WG_XY_SIZE = 2;
layout(constant_id = 1) const int WG_C_SIZE = 32;
layout(constant_id = 2) const int C_PER_THREAD = 1;
layout(constant_id = 3) const int USE_FP16_STORAGE = 0;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
  float epsilon;
} pc;

layout(binding = 0) readonly buffer InputBuf { float data[]; } inputBuf;
layout(binding = 1) writeonly buffer OutputBuf { float data[]; } outputBuf;
layout(binding = 2) readonly buffer WeightBuf { float data[]; } weightBuf;
layout(binding = 3) readonly buffer BetaBuf { float data[]; } betaBuf;
layout(binding = 4) readonly buffer MaskBuf { float data[]; } maskBuf;
layout(binding = 5) readonly buffer InputBufH { float16_t data[]; } inputBufH;
layout(binding = 6) writeonly buffer OutBufH { float16_t data[]; } outBufH;
layout(binding = 7) readonly buffer MaskBufH { float16_t data[]; } maskBufH;

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
  // The Vulkan spec is explicit that there is no direct relationship between
  // gl_LocalInvocationID and gl_SubgroupInvocationID, and that with full
  // subgroups an application should construct its own dense index as
  //   index = SubgroupLocalInvocationId + SubgroupId * SubgroupSize
  // (vkspec, interfaces-builtin-variables-sgli). Using it here is what makes
  // the WG_C_SIZE-sized group of `lid` values sit contiguously in subgroup-lane
  // space, so subgroupShuffleXor and the cross-group broadcast below are well
  // defined. The host guarantees this shader runs only with requireFullSubgroups
  // + a fixed subgroup size (see canUseRMSNormSubgroupVariant) so this
  // index is dense over [0, local_size_x).
  const int lid = int(gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID);
  int lidXY = lid / WG_C_SIZE;
  int lidC = lid % WG_C_SIZE;
  int xy = int(gl_WorkGroupID.x) * WG_XY_SIZE + lidXY;
  int n = int(gl_GlobalInvocationID.y);
  float maskVal = xy < pc.paddedSpatialSize ? ldMask(n * pc.paddedSpatialSize + xy) : 0.0;

  float acc = 0.0;
  for(int base = lidC * C_PER_THREAD; base < pc.numChannels; base += WG_C_SIZE * C_PER_THREAD) {
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < pc.numChannels && xy < pc.paddedSpatialSize) {
        float val = ldIn((n * pc.paddedSpatialSize + xy) * pc.numChannels + c) * maskVal;
        acc += val * val;
      }
    }
  }
  for(int span = WG_C_SIZE / 2; span > 0; span >>= 1)
    acc += subgroupShuffleXor(acc, span);
  int lane = int(gl_SubgroupInvocationID);
  float reduced = subgroupShuffle(acc, (lane / WG_C_SIZE) * WG_C_SIZE);
  float rms = inversesqrt(reduced / float(pc.numChannels) + pc.epsilon);

  for(int base = lidC * C_PER_THREAD; base < pc.numChannels; base += WG_C_SIZE * C_PER_THREAD) {
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < pc.numChannels && xy < pc.paddedSpatialSize) {
        int idx = (n * pc.paddedSpatialSize + xy) * pc.numChannels + c;
        stOut(idx, (ldIn(idx) * rms * weightBuf.data[c] + betaBuf.data[c]) * maskVal);
      }
    }
  }
}

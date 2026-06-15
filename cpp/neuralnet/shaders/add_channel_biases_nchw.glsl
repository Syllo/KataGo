// GLSL Compute Shader: Add per-(batch, channel) bias to every spatial position of an NCHW tensor
// Ports OpenCLKernels::addChannelBiasesNCHW
// Bias is ALWAYS FP32 (comes from MatMul output or fixed weights).
// Tensor may be FP16 (USE_FP16_STORAGE=1) or FP32.
// Dispatch: global(ceil(paddedSpatialSize/64), batchChannelCount, 1)

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
  int batchChannelCount;  // batchSize * numChannels (flat)
  int paddedSpatialSize;  // padded nnXLen*nnYLen
}
pc;

// FP32 path
layout(binding = 0) buffer TensorBuf {
  float data[];
}
tensorBuf;  // [batchChannelCount, paddedSpatialSize]
layout(binding = 1) readonly buffer BiasBuf {
  float data[];
}
biasBuf;  // [batchChannelCount] always FP32
// FP16 path (tensor only, bias stays FP32)
layout(binding = 2) buffer TensorBufH {
  float16_t data[];
}
tensorBufH;

float ldT(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(tensorBufH.data[idx])) : tensorBuf.data[idx];
}
void stT(int idx, float v) {
  if(USE_FP16_STORAGE == 1)
    tensorBufH.data[idx] = float16_t(v);
  else
    tensorBuf.data[idx] = v;
}

void main() {
  int xy = int(gl_GlobalInvocationID.x);
  int nc = int(gl_GlobalInvocationID.y);
  if(xy >= pc.paddedSpatialSize || nc >= pc.batchChannelCount)
    return;

  int idx = nc * pc.paddedSpatialSize + xy;
  stT(idx, ldT(idx) + biasBuf.data[nc]);  // bias always FP32
}

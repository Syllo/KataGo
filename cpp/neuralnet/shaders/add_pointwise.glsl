// GLSL Compute Shader: Element-wise add (residual skip connection)
// Ports OpenCLKernels::addPointWise
// acc += value  (same size)
// Dispatch: global(numElements, 1, 1)

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
  int numElements;
}
pc;

layout(binding = 0) buffer AccBuf {
  float data[];
}
accBuf;
layout(binding = 1) readonly buffer ValBuf {
  float data[];
}
valBuf;
layout(binding = 2) buffer AccBufH {
  float16_t data[];
}
accBufH;
layout(binding = 3) readonly buffer ValBufH {
  float16_t data[];
}
valBufH;

void main() {
  int i = int(gl_GlobalInvocationID.x);
  if(i >= pc.numElements)
    return;

  if(USE_FP16_STORAGE == 1) {
    float a = float(float16_t(accBufH.data[i]));
    float v = float(float16_t(valBufH.data[i]));
    accBufH.data[i] = float16_t(a + v);
  } else {
    accBuf.data[i] += valBuf.data[i];
  }
}

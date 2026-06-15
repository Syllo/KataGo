// GLSL Compute Shader: SwiGLU activation for transformer FFN
// Ports OpenCLKernels::transformerSwiGLU
// output[i] = silu(mainProj[i]) * gateProj[i]
// silu(x) = x / (1 + exp(-x))
// Dispatch: global(numElements/4, 1, 1)

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

layout(binding = 0) readonly buffer MainBuf {
  vec4 data[];
}
mainBuf;
layout(binding = 1) readonly buffer GateBuf {
  vec4 data[];
}
gateBuf;
layout(binding = 2) writeonly buffer OutBuf {
  vec4 data[];
}
outBuf;
layout(binding = 3) readonly buffer MainBufH {
  f16vec4 data[];
}
mainBufH;
layout(binding = 4) readonly buffer GateBufH {
  f16vec4 data[];
}
gateBufH;
layout(binding = 5) writeonly buffer OutBufH {
  f16vec4 data[];
}
outBufH;

vec4 ldM4(int idx) {
  return USE_FP16_STORAGE == 1 ? vec4(mainBufH.data[idx]) : mainBuf.data[idx];
}
vec4 ldG4(int idx) {
  return USE_FP16_STORAGE == 1 ? vec4(gateBufH.data[idx]) : gateBuf.data[idx];
}
void stO4(int idx, vec4 v) {
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = f16vec4(v);
  else
    outBuf.data[idx] = v;
}

void main() {
  int i4 = int(gl_GlobalInvocationID.x);
  int numVec4 = pc.numElements / 4;
  if(i4 >= numVec4)
    return;

  vec4 a = ldM4(i4);
  vec4 g = ldG4(i4);
  vec4 silu_a = a / (vec4(1.0) + exp(-a));
  stO4(i4, silu_a * g);
}

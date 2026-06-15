// GLSL Compute Shader: Add bias + activation for MatBias (NC format)
// Ports OpenCLKernels::addCBiasesNCAct
// Dispatch: global(numChannels, batchSize, 1)

#version 450

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int ACTIVATION = 0;  // 0=iden,1=relu,2=mish,3=silu,12=mish_scale8
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
}
pc;

layout(binding = 0) buffer TensorBuf {
  float data[];
}
tensorBuf;  // [N, C]
layout(binding = 1) readonly buffer BiasBuf {
  float data[];
}
biasBuf;  // [C]

float applyAct(float a) {
  if(ACTIVATION == 0)
    return a;
  if(ACTIVATION == 1)
    return max(a, 0.0);
  if(ACTIVATION == 2) {
    float sp = a < 20.0 ? log(1.0 + exp(a)) : a;
    return a * tanh(sp);
  }
  if(ACTIVATION == 12) {
    return a < 2.5 ? a * tanh(log(1.0 + exp(a * 8.0))) : a;
  }  // mish_scale8 (matches CUDA/OpenCL)
  if(ACTIVATION == 3)
    return a / (1.0 + exp(-a));
  return a;
}

void main() {
  int c = int(gl_GlobalInvocationID.x);
  int n = int(gl_GlobalInvocationID.y);
  if(c >= pc.numChannels)
    return;

  int idx = n * pc.numChannels + c;
  float a = tensorBuf.data[idx] + biasBuf.data[c];
  tensorBuf.data[idx] = applyAct(a);
}

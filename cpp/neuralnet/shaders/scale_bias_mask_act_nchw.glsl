// GLSL Compute Shader: BatchNorm apply: scale * x + bias + activation, masked
// Ports OpenCLKernels::scaleBiasMaskActNCHW
// Dispatch: global(paddedSpatialSize, numChannels, 1), iterates over batchSize in loop

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int ACTIVATION = 1;  // 0=iden,1=relu,2=mish,3=silu,12=mish_scale8
layout(constant_id = 1) const int USE_FP16_STORAGE = 0;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numChannels;
  int paddedSpatialSize;
  int batchSize;
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
layout(binding = 2) readonly buffer ScaleBuf {
  float data[];
}
scaleBuf;
layout(binding = 3) readonly buffer BiasBuf {
  float data[];
}
biasBuf;
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
layout(binding = 7) readonly buffer ScaleBufH {
  float16_t data[];
}
scaleBufH;
layout(binding = 8) readonly buffer BiasBufH {
  float16_t data[];
}
biasBufH;
layout(binding = 9) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

float loadF(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}
float loadS(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(scaleBufH.data[idx])) : scaleBuf.data[idx];
}
float loadB(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(biasBufH.data[idx])) : biasBuf.data[idx];
}
float loadM(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}
void storeO(int idx, float v) {
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = float16_t(v);
  else
    outputBuf.data[idx] = v;
}

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
    // mish_scale8: a < 2.5 ? a*tanh(softplus(8a)) : a   (matches CUDA/OpenCL)
    return a < 2.5 ? a * tanh(log(1.0 + exp(a * 8.0))) : a;
  }
  if(ACTIVATION == 3)
    return a / (1.0 + exp(-a));
  return a;
}

void main() {
  int xy = int(gl_GlobalInvocationID.x);
  int c = int(gl_GlobalInvocationID.y);
  if(xy >= pc.paddedSpatialSize || c >= pc.numChannels)
    return;

  float s = loadS(c);
  float b = loadB(c);

  for(int n = 0; n < pc.batchSize; n++) {
    int idx = (n * pc.numChannels + c) * pc.paddedSpatialSize + xy;
    float a = loadF(idx) * s + b;
    float m = loadM(n * pc.paddedSpatialSize + xy);
    storeO(idx, applyAct(a) * m);
  }
}

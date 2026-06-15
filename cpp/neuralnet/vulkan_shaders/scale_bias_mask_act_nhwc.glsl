// GLSL Compute Shader: BatchNorm scale/bias/mask/activation for NHWC tensors.
// input/output: [N, paddedSpatialSize, C]. Arithmetic is FP32 regardless of
// storage; USE_FP16_STORAGE controls only loads and stores.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int ACTIVATION = 1;  // 0=iden,1=relu,2=mish,3=silu,12=mish_scale8
layout(constant_id = 1) const int USE_FP16_STORAGE = 0;
layout(constant_id = 2) const int USE_VEC4 = 0;
layout(constant_id = 3) const int HAS_DYNAMIC_BIAS = 0;

// X tiles contiguous channels and Y tiles four spatial positions. Z is batch.
layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;

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
// Optional FP32 [N,C] bias produced by a global-pooling branch. Keep this
// before the storage-format aliases so its binding exists in every pipeline.
layout(binding = 5) readonly buffer DynamicBiasBuf {
  float data[];
}
dynamicBiasBuf;
layout(binding = 5) readonly buffer DynamicBiasBufV4 {
  vec4 data[];
}
dynamicBiasBufV4;

layout(binding = 6) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 7) writeonly buffer OutputBufH {
  float16_t data[];
}
outputBufH;
layout(binding = 8) readonly buffer ScaleBufH {
  float16_t data[];
}
scaleBufH;
layout(binding = 9) readonly buffer BiasBufH {
  float16_t data[];
}
biasBufH;
layout(binding = 10) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

// Vector aliases of the activation and per-channel buffers. Each active
// pipeline binds these descriptors to the same buffers as the scalar views.
layout(binding = 11) readonly buffer InputBufV4 {
  vec4 data[];
}
inputBufV4;
layout(binding = 12) writeonly buffer OutputBufV4 {
  vec4 data[];
}
outputBufV4;
layout(binding = 13) readonly buffer ScaleBufV4 {
  vec4 data[];
}
scaleBufV4;
layout(binding = 14) readonly buffer BiasBufV4 {
  vec4 data[];
}
biasBufV4;
layout(binding = 15) readonly buffer InputBufV4H {
  f16vec4 data[];
}
inputBufV4H;
layout(binding = 16) writeonly buffer OutputBufV4H {
  f16vec4 data[];
}
outputBufV4H;
layout(binding = 17) readonly buffer ScaleBufV4H {
  f16vec4 data[];
}
scaleBufV4H;
layout(binding = 18) readonly buffer BiasBufV4H {
  f16vec4 data[];
}
biasBufV4H;

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
    outputBufH.data[idx] = float16_t(v);
  else
    outputBuf.data[idx] = v;
}

vec4 loadF4(int idx) {
  return USE_FP16_STORAGE == 1 ? vec4(inputBufV4H.data[idx]) : inputBufV4.data[idx];
}
vec4 loadS4(int idx) {
  return USE_FP16_STORAGE == 1 ? vec4(scaleBufV4H.data[idx]) : scaleBufV4.data[idx];
}
vec4 loadB4(int idx) {
  return USE_FP16_STORAGE == 1 ? vec4(biasBufV4H.data[idx]) : biasBufV4.data[idx];
}
void storeO4(int idx, vec4 v) {
  if(USE_FP16_STORAGE == 1)
    outputBufV4H.data[idx] = f16vec4(v);
  else
    outputBufV4.data[idx] = v;
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
  if(ACTIVATION == 12)
    return a < 2.5 ? a * tanh(log(1.0 + exp(a * 8.0))) : a;
  if(ACTIVATION == 3)
    return a / (1.0 + exp(-a));
  return a;
}

void main() {
  int channelLane = int(gl_GlobalInvocationID.x);
  int xy = int(gl_GlobalInvocationID.y);
  int n = int(gl_GlobalInvocationID.z);
  if(xy >= pc.paddedSpatialSize || n >= pc.batchSize)
    return;

  float m = loadM(n * pc.paddedSpatialSize + xy);
  if(USE_VEC4 == 1) {
    int numChannelPacks = pc.numChannels / 4;
    if(channelLane >= numChannelPacks)
      return;
    int idx4 = (n * pc.paddedSpatialSize + xy) * numChannelPacks + channelLane;
    vec4 a = loadF4(idx4);
    if(HAS_DYNAMIC_BIAS == 1)
      a += dynamicBiasBufV4.data[n * numChannelPacks + channelLane];
    a = a * loadS4(channelLane) + loadB4(channelLane);
    vec4 activated = vec4(applyAct(a.x), applyAct(a.y), applyAct(a.z), applyAct(a.w));
    storeO4(idx4, activated * m);
  } else {
    int c = channelLane;
    if(c >= pc.numChannels)
      return;
    int idx = (n * pc.paddedSpatialSize + xy) * pc.numChannels + c;
    float a = loadF(idx);
    if(HAS_DYNAMIC_BIAS == 1)
      a += dynamicBiasBuf.data[n * pc.numChannels + c];
    a = a * loadS(c) + loadB(c);
    storeO(idx, applyAct(a) * m);
  }
}

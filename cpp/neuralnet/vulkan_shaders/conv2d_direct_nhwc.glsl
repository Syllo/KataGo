// Generic direct 2-D convolution on NHWC activations.
// input: [N, paddedSpatialSize, ic]  filter: [oc, ic, fy, fx]
// output: [N, paddedSpatialSize, oc]. Arithmetic is FP32 in both storage modes.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int TILE_XSIZE = 8;
layout(constant_id = 1) const int TILE_YSIZE = 4;
layout(constant_id = 2) const int TILE_CHANNELS = 4;
layout(constant_id = 3) const int USE_FP16_STORAGE = 0;

layout(local_size_x_id = 0) in;
layout(local_size_y_id = 1) in;
layout(local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int nnXLen;
  int nnYLen;
  int numOutChannels;
  int numInChannels;
  int filterXRadius;
  int filterYRadius;
  int paddedSpatialSize;
  int batchSize;
}
pc;

layout(binding = 0) readonly buffer InputBuf {
  float data[];
}
inputBuf;
layout(binding = 1) readonly buffer FilterBuf {
  float data[];
}
filterBuf;
layout(binding = 2) writeonly buffer OutputBuf {
  float data[];
}
outputBuf;
layout(binding = 3) readonly buffer InputBufH {
  float16_t data[];
}
inputBufH;
layout(binding = 4) writeonly buffer OutBufH {
  float16_t data[];
}
outBufH;
layout(binding = 5) readonly buffer FilterBufH {
  float16_t data[];
}
filterBufH;

float ldIn(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}
float ldFilter(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(filterBufH.data[idx])) : filterBuf.data[idx];
}
void stOut(int idx, float v) {
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = float16_t(v);
  else
    outputBuf.data[idx] = v;
}

void main() {
  int ox = int(gl_GlobalInvocationID.x);
  int oy = int(gl_GlobalInvocationID.y);
  int oc = int(gl_GlobalInvocationID.z);
  if(ox >= pc.nnXLen || oy >= pc.nnYLen || oc >= pc.numOutChannels)
    return;

  int fxSize = pc.filterXRadius * 2 + 1;
  int fySize = pc.filterYRadius * 2 + 1;
  for(int n = 0; n < pc.batchSize; n++) {
    float acc = 0.0;
    for(int ic = 0; ic < pc.numInChannels; ic++) {
      for(int fy = 0; fy < fySize; fy++) {
        int iy = oy + fy - pc.filterYRadius;
        if(iy < 0 || iy >= pc.nnYLen)
          continue;
        for(int fx = 0; fx < fxSize; fx++) {
          int ix = ox + fx - pc.filterXRadius;
          if(ix < 0 || ix >= pc.nnXLen)
            continue;
          int xy = iy * pc.nnXLen + ix;
          int inIdx = (n * pc.paddedSpatialSize + xy) * pc.numInChannels + ic;
          int filIdx = ((oc * pc.numInChannels + ic) * fySize + fy) * fxSize + fx;
          acc += ldIn(inIdx) * ldFilter(filIdx);
        }
      }
    }
    int outXY = oy * pc.nnXLen + ox;
    int outIdx = (n * pc.paddedSpatialSize + outXY) * pc.numOutChannels + oc;
    stOut(outIdx, acc);
  }
}

// Shared-memory transpose of a spatial tensor from NCHW [N,C,XY] to
// NHWC [N,XY,C]. Loads and stores are contiguous on opposite sides of the
// transpose, and the padded shared stride avoids bank conflicts.
#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(constant_id = 0) const int USE_FP16_STORAGE = 0;
layout(constant_id = 1) const uint TILE_DIM = 32u;
layout(local_size_x_id = 2, local_size_y = 8, local_size_z = 1) in;

const uint BLOCK_ROWS = 8u;

layout(push_constant) uniform PushConstants {
  int inputChannels;
  int outputChannels;
  int paddedSpatialSize;
  int batchSize;
} pc;

layout(binding = 0) readonly buffer InputBuf { float data[]; } inputBuf;
layout(binding = 1) writeonly buffer OutputBuf { float data[]; } outputBuf;
layout(binding = 2) readonly buffer InputBufH { float16_t data[]; } inputBufH;
layout(binding = 3) writeonly buffer OutputBufH { float16_t data[]; } outputBufH;

shared float tile[TILE_DIM][TILE_DIM + 1u];

float loadInput(uint idx) {
  return USE_FP16_STORAGE == 1 ? float(inputBufH.data[idx]) : inputBuf.data[idx];
}

void storeOutput(uint idx, float value) {
  if(USE_FP16_STORAGE == 1)
    outputBufH.data[idx] = float16_t(value);
  else
    outputBuf.data[idx] = value;
}

void main() {
  const uint inputChannels = uint(pc.inputChannels);
  const uint outputChannels = uint(pc.outputChannels);
  const uint spatialSize = uint(pc.paddedSpatialSize);
  const uint localX = gl_LocalInvocationID.x;
  const uint localY = gl_LocalInvocationID.y;
  const uint channelBase = gl_WorkGroupID.x * TILE_DIM;
  const uint spatialBase = gl_WorkGroupID.y * TILE_DIM;
  const uint n = gl_WorkGroupID.z;

  for(uint j = 0u; j < TILE_DIM; j += BLOCK_ROWS) {
    const uint c = channelBase + localY + j;
    const uint xy = spatialBase + localX;
    if(c < outputChannels && xy < spatialSize)
      tile[localY + j][localX] = c < inputChannels
        ? loadInput((n * inputChannels + c) * spatialSize + xy)
        : 0.0;
  }

  barrier();

  for(uint j = 0u; j < TILE_DIM; j += BLOCK_ROWS) {
    const uint xy = spatialBase + localY + j;
    const uint c = channelBase + localX;
    if(xy < spatialSize && c < outputChannels)
      storeOutput((n * spatialSize + xy) * outputChannels + c, tile[localX][localY + j]);
  }
}

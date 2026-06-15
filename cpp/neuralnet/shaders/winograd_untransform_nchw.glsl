// GLSL Compute Shader: Winograd output tile untransform
// Ports OpenCLKernels::winogradUntransformNCHW
// Dispatch: global(numTilesX, numTilesY, batchSize*numOutChannels)

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int INTILE_XSIZE = 4;
layout(constant_id = 1) const int INTILE_YSIZE = 4;
layout(constant_id = 2) const int OUTTILE_XSIZE = 2;
layout(constant_id = 3) const int OUTTILE_YSIZE = 2;
layout(constant_id = 4) const int FILTER_XSIZE = 3;
layout(constant_id = 5) const int FILTER_YSIZE = 3;
layout(constant_id = 6) const int USE_FP16_STORAGE = 0;
layout(constant_id = 7) const int LOCAL_SIZE_X = 8;
layout(constant_id = 8) const int LOCAL_SIZE_Y = 8;
layout(constant_id = 9) const int ADD_TO_OUTPUT = 0;

layout(local_size_x_id = 7, local_size_y_id = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int nnXLen;
  int nnYLen;
  int numTilesX;
  int numTilesY;
  int numOutChannels;
  int numOutChannelsPadded;
  int numTilesPadded;
  int paddedSpatialSize;
}
pc;

layout(binding = 0) readonly buffer TransBuf {
  float data[];
}
transBuf;
layout(binding = 1) buffer OutputBuf {
  float data[];
}
outputBuf;
layout(binding = 2) readonly buffer TransBufH {
  float16_t data[];
}
transBufH;
layout(binding = 3) buffer OutputBufH {
  float16_t data[];
}
outputBufH;

float loadTrans(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(transBufH.data[idx])) : transBuf.data[idx];
}
float loadOutput(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(outputBufH.data[idx])) : outputBuf.data[idx];
}
void storeOutput(int idx, float val) {
  if(ADD_TO_OUTPUT != 0)
    val += loadOutput(idx);
  if(USE_FP16_STORAGE == 1)
    outputBufH.data[idx] = float16_t(val);
  else
    outputBuf.data[idx] = val;
}

void main() {
  int tileX = int(gl_GlobalInvocationID.x);
  int tileY = int(gl_GlobalInvocationID.y);
  int noc = int(gl_GlobalInvocationID.z);
  int n = noc / pc.numOutChannels;
  int oc = noc % pc.numOutChannels;

  if(tileX >= pc.numTilesX || tileY >= pc.numTilesY)
    return;

  int ntile = (n * pc.numTilesY + tileY) * pc.numTilesX + tileX;

  float wTile[36];

  int transBase = oc * pc.numTilesPadded + ntile;
  int transTileStride = pc.numOutChannelsPadded * pc.numTilesPadded;
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int winoTile = subY * INTILE_XSIZE + subX;
      wTile[subY * INTILE_XSIZE + subX] = loadTrans(transBase + winoTile * transTileStride);
    }
  }

  // X untransform
  if(FILTER_XSIZE == 3 && OUTTILE_XSIZE == 2) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0], z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2], z3 = wTile[subY * INTILE_XSIZE + 3];
      wTile[subY * INTILE_XSIZE + 0] = z0 + z1 + z2;
      wTile[subY * INTILE_XSIZE + 1] = z1 - z2 - z3;
    }
  } else if(FILTER_XSIZE == 3 && OUTTILE_XSIZE == 4) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0], z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2], z3 = wTile[subY * INTILE_XSIZE + 3];
      float z4 = wTile[subY * INTILE_XSIZE + 4], z5 = wTile[subY * INTILE_XSIZE + 5];
      wTile[subY * INTILE_XSIZE + 0] = z0 + z1 + z2 + z3 + z4;
      wTile[subY * INTILE_XSIZE + 1] = (z1 - z2) + 2.0 * (z3 - z4);
      wTile[subY * INTILE_XSIZE + 2] = (z1 + z2) + 4.0 * (z3 + z4);
      wTile[subY * INTILE_XSIZE + 3] = (z1 - z2) + 8.0 * (z3 - z4) + z5;
    }
  } else if(FILTER_XSIZE == 5 && OUTTILE_XSIZE == 2) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0], z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2], z3 = wTile[subY * INTILE_XSIZE + 3];
      float z4 = wTile[subY * INTILE_XSIZE + 4], z5 = wTile[subY * INTILE_XSIZE + 5];
      wTile[subY * INTILE_XSIZE + 0] = z0 + z1 + z2 + z3 + z4;
      wTile[subY * INTILE_XSIZE + 1] = (z1 - z2) + 2.0 * (z3 - z4) + z5;
    }
  }

  // Y untransform
  if(FILTER_YSIZE == 3 && OUTTILE_YSIZE == 2) {
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX], z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX], z3 = wTile[3 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = z0 + z1 + z2;
      wTile[1 * INTILE_XSIZE + subX] = z1 - z2 - z3;
    }
  } else if(FILTER_YSIZE == 3 && OUTTILE_YSIZE == 4) {
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX], z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX], z3 = wTile[3 * INTILE_XSIZE + subX];
      float z4 = wTile[4 * INTILE_XSIZE + subX], z5 = wTile[5 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = z0 + z1 + z2 + z3 + z4;
      wTile[1 * INTILE_XSIZE + subX] = (z1 - z2) + 2.0 * (z3 - z4);
      wTile[2 * INTILE_XSIZE + subX] = (z1 + z2) + 4.0 * (z3 + z4);
      wTile[3 * INTILE_XSIZE + subX] = (z1 - z2) + 8.0 * (z3 - z4) + z5;
    }
  } else if(FILTER_YSIZE == 5 && OUTTILE_YSIZE == 2) {
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX], z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX], z3 = wTile[3 * INTILE_XSIZE + subX];
      float z4 = wTile[4 * INTILE_XSIZE + subX], z5 = wTile[5 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = z0 + z1 + z2 + z3 + z4;
      wTile[1 * INTILE_XSIZE + subX] = (z1 - z2) + 2.0 * (z3 - z4) + z5;
    }
  }

  for(int subY = 0; subY < OUTTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY;
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX;
      if(y < pc.nnYLen && x < pc.nnXLen) {
        int outIdx = noc * pc.paddedSpatialSize + y * pc.nnXLen + x;
        storeOutput(outIdx, wTile[subY * INTILE_XSIZE + subX]);
      }
    }
  }
}

// GLSL Compute Shader: Winograd input tile transform (no fused BN)
// Ports OpenCLKernels::winogradTransformNCHW
// Specialization constants select conv/tile sizes and FP16 mode.
//
// Dispatch: global(numInChannelsPadded, numTilesPadded, 1)
// Each invocation handles one (ic, ntxty) pair.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

// Specialization constants
layout(constant_id = 0) const int INTILE_XSIZE = 4;  // 4 for 3x3, 6 for 5x5
layout(constant_id = 1) const int INTILE_YSIZE = 4;
layout(constant_id = 2) const int OUTTILE_XSIZE = 2;  // 2 for 3x3, 2 for 5x5
layout(constant_id = 3) const int OUTTILE_YSIZE = 2;
layout(constant_id = 4) const int FILTER_XSIZE = 3;  // 3 or 5
layout(constant_id = 5) const int FILTER_YSIZE = 3;
layout(constant_id = 6) const int INTILE_XOFFSET = -1;  // -1 for 3x3, -2 for 5x5
layout(constant_id = 7) const int INTILE_YOFFSET = -1;
layout(constant_id = 8) const int USE_FP16_STORAGE = 0;  // 1 = FP16 input/output
layout(constant_id = 9) const int LOCAL_SIZE_X = 8;
layout(constant_id = 10) const int LOCAL_SIZE_Y = 8;
layout(constant_id = 11) const int PACKED_A_BM = 64;
layout(constant_id = 12) const int PACKED_A_BK = 16;
layout(constant_id = 13) const int PACKED_A_PAD_WORDS = 1;

layout(local_size_x_id = 9, local_size_y_id = 10, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int nnXLen;
  int nnYLen;
  int numTilesX;
  int numTilesY;
  int numInChannels;
  int numInChannelsPadded;
  int numTilesReal;
  int numTilesPadded;
  int paddedSpatialSize;
}
pc;

layout(binding = 0) readonly buffer InputBuf {
  float data[];
}
inputBuf;
layout(binding = 1) writeonly buffer TransBuf {
  float data[];
}
transBuf;
layout(binding = 2) readonly buffer InputBufHalf {
  float16_t data[];
}
inputBufHalf;
layout(binding = 3) writeonly buffer TransBufHalf {
  float16_t data[];
}
transBufHalf;

float loadInput(int nic_xy) {
  if(USE_FP16_STORAGE == 1)
    return float(inputBufHalf.data[nic_xy]);
  else
    return inputBuf.data[nic_xy];
}

void storeTrans(int idx, float val) {
  if(USE_FP16_STORAGE == 1)
    transBufHalf.data[idx] = float16_t(val);
  else
    transBuf.data[idx] = val;
}

void transformedStoreBaseStride(int ic, int ntxty, out int base, out int tileStride) {
  int mTile = ntxty / PACKED_A_BM;
  int mInTile = ntxty - mTile * PACKED_A_BM;
  int kBlock = ic / PACKED_A_BK;
  int kk = ic - kBlock * PACKED_A_BK;
  int kkWord = kk / 4;
  int kkLane = kk - kkWord * 4;
  int mTiles = (pc.numTilesPadded + PACKED_A_BM - 1) / PACKED_A_BM;
  int kBlocks = (pc.numInChannelsPadded + PACKED_A_BK - 1) / PACKED_A_BK;
  int rowWords = PACKED_A_BK / 4 + PACKED_A_PAD_WORDS;
  base = (((mTile * kBlocks + kBlock) * PACKED_A_BM + mInTile) * rowWords + kkWord) * 4 + kkLane;
  tileStride = mTiles * kBlocks * PACKED_A_BM * rowWords * 4;
}

void storeTransformedTile(int base, int tileStride, int winoTile, float val) {
  storeTrans(base + winoTile * tileStride, val);
}

void main() {
  int ic = int(gl_GlobalInvocationID.x);
  int ntxty = int(gl_GlobalInvocationID.y);

  if(ntxty >= pc.numTilesPadded || ic >= pc.numInChannelsPadded)
    return;
  bool validInput = ntxty < pc.numTilesReal && ic < pc.numInChannels;

  int tileX = ntxty % pc.numTilesX;
  int tmp = ntxty / pc.numTilesX;
  int tileY = tmp % pc.numTilesY;
  int n = tmp / pc.numTilesY;
  int nic = n * pc.numInChannels + ic;

  // Private tile: max 6x6 = 36 elements
  float wTile[36];
  // Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      float value = 0.0;
      if(validInput && y >= 0 && y < pc.nnYLen && x >= 0 && x < pc.nnXLen) {
        int xy = y * pc.nnXLen + x;
        value = loadInput(nic * pc.paddedSpatialSize + xy);
      }
      wTile[subY * INTILE_XSIZE + subX] = value;
    }
  }

  // X transform
  if(FILTER_XSIZE == 3 && OUTTILE_XSIZE == 2) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0];
      float z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2];
      float z3 = wTile[subY * INTILE_XSIZE + 3];
      wTile[subY * INTILE_XSIZE + 0] = z0 - z2;
      wTile[subY * INTILE_XSIZE + 1] = z1 + z2;
      wTile[subY * INTILE_XSIZE + 2] = z2 - z1;
      wTile[subY * INTILE_XSIZE + 3] = z1 - z3;
    }
  } else if(FILTER_XSIZE == 3 && OUTTILE_XSIZE == 4) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0];
      float z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2];
      float z3 = wTile[subY * INTILE_XSIZE + 3];
      float z4 = wTile[subY * INTILE_XSIZE + 4];
      float z5 = wTile[subY * INTILE_XSIZE + 5];
      wTile[subY * INTILE_XSIZE + 0] = 4.0 * z0 - 5.0 * z2 + z4;
      wTile[subY * INTILE_XSIZE + 1] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
      wTile[subY * INTILE_XSIZE + 2] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
      wTile[subY * INTILE_XSIZE + 3] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
      wTile[subY * INTILE_XSIZE + 4] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
      wTile[subY * INTILE_XSIZE + 5] = 4.0 * z1 - 5.0 * z3 + z5;
    }
  } else if(FILTER_XSIZE == 5 && OUTTILE_XSIZE == 2) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0];
      float z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2];
      float z3 = wTile[subY * INTILE_XSIZE + 3];
      float z4 = wTile[subY * INTILE_XSIZE + 4];
      float z5 = wTile[subY * INTILE_XSIZE + 5];
      wTile[subY * INTILE_XSIZE + 0] = 4.0 * z0 - 5.0 * z2 + z4;
      wTile[subY * INTILE_XSIZE + 1] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
      wTile[subY * INTILE_XSIZE + 2] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
      wTile[subY * INTILE_XSIZE + 3] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
      wTile[subY * INTILE_XSIZE + 4] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
      wTile[subY * INTILE_XSIZE + 5] = 4.0 * z1 - 5.0 * z3 + z5;
    }
  }

  // Y transform
  if(FILTER_YSIZE == 3 && OUTTILE_YSIZE == 2) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX];
      float z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX];
      float z3 = wTile[3 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = z0 - z2;
      wTile[1 * INTILE_XSIZE + subX] = z1 + z2;
      wTile[2 * INTILE_XSIZE + subX] = z2 - z1;
      wTile[3 * INTILE_XSIZE + subX] = z1 - z3;
    }
  } else if(FILTER_YSIZE == 3 && OUTTILE_YSIZE == 4) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX];
      float z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX];
      float z3 = wTile[3 * INTILE_XSIZE + subX];
      float z4 = wTile[4 * INTILE_XSIZE + subX];
      float z5 = wTile[5 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = 4.0 * z0 - 5.0 * z2 + z4;
      wTile[1 * INTILE_XSIZE + subX] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
      wTile[2 * INTILE_XSIZE + subX] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
      wTile[3 * INTILE_XSIZE + subX] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
      wTile[4 * INTILE_XSIZE + subX] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
      wTile[5 * INTILE_XSIZE + subX] = 4.0 * z1 - 5.0 * z3 + z5;
    }
  } else if(FILTER_YSIZE == 5 && OUTTILE_YSIZE == 2) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX];
      float z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX];
      float z3 = wTile[3 * INTILE_XSIZE + subX];
      float z4 = wTile[4 * INTILE_XSIZE + subX];
      float z5 = wTile[5 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = 4.0 * z0 - 5.0 * z2 + z4;
      wTile[1 * INTILE_XSIZE + subX] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
      wTile[2 * INTILE_XSIZE + subX] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
      wTile[3 * INTILE_XSIZE + subX] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
      wTile[4 * INTILE_XSIZE + subX] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
      wTile[5 * INTILE_XSIZE + subX] = 4.0 * z1 - 5.0 * z3 + z5;
    }
  }

  // Write row-major packed A: [winoTile, M-tile, K-block, M, K/4+pad].
  int transBase;
  int transTileStride;
  transformedStoreBaseStride(ic, ntxty, transBase, transTileStride);
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int winoTile = subY * INTILE_XSIZE + subX;
      storeTransformedTile(transBase, transTileStride, winoTile, wTile[subY * INTILE_XSIZE + subX]);
    }
  }
}

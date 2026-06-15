// GLSL Compute Shader: Winograd input tile transform with fused BatchNorm + Activation
// Ports OpenCLKernels::winogradBNActTransformNCHW
// Specialization constants select conv/tile sizes, activation, and FP16 mode.
//
// Dispatch: global(numInChannelsPadded, numTilesPadded, 1)

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
layout(constant_id = 6) const int INTILE_XOFFSET = -1;
layout(constant_id = 7) const int INTILE_YOFFSET = -1;
layout(constant_id = 8) const int ACTIVATION = 1;  // 0=iden,1=relu,2=mish,3=silu,12=mish_scale8
layout(constant_id = 9) const int USE_FP16_STORAGE = 0;
layout(constant_id = 10) const int LOCAL_SIZE_X = 8;
layout(constant_id = 11) const int LOCAL_SIZE_Y = 8;
layout(constant_id = 12) const int PACKED_A_BM = 64;
layout(constant_id = 13) const int PACKED_A_BK = 16;
layout(constant_id = 14) const int PACKED_A_PAD_WORDS = 1;

layout(local_size_x_id = 10, local_size_y_id = 11, local_size_z = 1) in;

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
layout(binding = 6) writeonly buffer TransBufH {
  float16_t data[];
}
transBufH;
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

float loadInput(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(inputBufH.data[idx])) : inputBuf.data[idx];
}
float loadScale(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(scaleBufH.data[idx])) : scaleBuf.data[idx];
}
float loadBias(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(biasBufH.data[idx])) : biasBuf.data[idx];
}
float loadMask(int idx) {
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}
void storeTrans(int idx, float val) {
  if(USE_FP16_STORAGE == 1)
    transBufH.data[idx] = float16_t(val);
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

float applyActivation(float a) {
  if(ACTIVATION == 0)
    return a;
  else if(ACTIVATION == 1)
    return max(a, 0.0);
  else if(ACTIVATION == 2) {
    float sp = a < 20.0 ? log(1.0 + exp(a)) : a;
    return a * tanh(sp);
  } else if(ACTIVATION == 12) {
    // mish_scale8: a < 2.5 ? a*tanh(softplus(8a)) : a   (matches CUDA/OpenCL)
    return a < 2.5 ? a * tanh(log(1.0 + exp(a * 8.0))) : a;
  } else if(ACTIVATION == 3) {
    return a / (1.0 + exp(-a));
  }
  return a;
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

  float wTile[36];

  float scaleVal = validInput ? loadScale(ic) : 0.0;
  float biasVal = validInput ? loadBias(ic) : 0.0;

  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      float value = 0.0;
      if(validInput && y >= 0 && y < pc.nnYLen && x >= 0 && x < pc.nnXLen) {
        int xy = y * pc.nnXLen + x;
        float maskVal = loadMask(n * pc.paddedSpatialSize + xy);
        float raw = loadInput(nic * pc.paddedSpatialSize + xy);
        float a = raw * scaleVal + biasVal;
        value = applyActivation(a) * maskVal;
      }
      wTile[subY * INTILE_XSIZE + subX] = value;
    }
  }

  // X transform
  if(FILTER_XSIZE == 3 && OUTTILE_XSIZE == 2) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0], z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2], z3 = wTile[subY * INTILE_XSIZE + 3];
      wTile[subY * INTILE_XSIZE + 0] = z0 - z2;
      wTile[subY * INTILE_XSIZE + 1] = z1 + z2;
      wTile[subY * INTILE_XSIZE + 2] = z2 - z1;
      wTile[subY * INTILE_XSIZE + 3] = z1 - z3;
    }
  } else if((FILTER_XSIZE == 3 && OUTTILE_XSIZE == 4) || (FILTER_XSIZE == 5 && OUTTILE_XSIZE == 2)) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      float z0 = wTile[subY * INTILE_XSIZE + 0], z1 = wTile[subY * INTILE_XSIZE + 1];
      float z2 = wTile[subY * INTILE_XSIZE + 2], z3 = wTile[subY * INTILE_XSIZE + 3];
      float z4 = wTile[subY * INTILE_XSIZE + 4], z5 = wTile[subY * INTILE_XSIZE + 5];
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
      float z0 = wTile[0 * INTILE_XSIZE + subX], z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX], z3 = wTile[3 * INTILE_XSIZE + subX];
      wTile[0 * INTILE_XSIZE + subX] = z0 - z2;
      wTile[1 * INTILE_XSIZE + subX] = z1 + z2;
      wTile[2 * INTILE_XSIZE + subX] = z2 - z1;
      wTile[3 * INTILE_XSIZE + subX] = z1 - z3;
    }
  } else if((FILTER_YSIZE == 3 && OUTTILE_YSIZE == 4) || (FILTER_YSIZE == 5 && OUTTILE_YSIZE == 2)) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      float z0 = wTile[0 * INTILE_XSIZE + subX], z1 = wTile[1 * INTILE_XSIZE + subX];
      float z2 = wTile[2 * INTILE_XSIZE + subX], z3 = wTile[3 * INTILE_XSIZE + subX];
      float z4 = wTile[4 * INTILE_XSIZE + subX], z5 = wTile[5 * INTILE_XSIZE + subX];
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

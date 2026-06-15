// GLSL Compute Shader: Winograd output tile untransform, NHWC layout
// Dispatch: global(numTilesX, numTilesY, batchSize*(numOutChannels/OC_VEC))
//
// The ADD_TO_OUTPUT residual-fusion path reads/writes the same recomputed address,
// so residual fusion works unchanged once the address formula is layout-correct.
//
// Channel-vectorized store (OC_VEC): the C-tensor load below is already
// unit-stride in `ntile` and well coalesced -- each thread does
// INTILE_XSIZE*INTILE_YSIZE loads (16 for F(2,3)) at a fixed `oc`, one per
// Winograd sub-tile. But the *store* is channel-strided: with `oc` on the
// slowest dispatch axis, adjacent lanes (which vary `tileX`) write addresses
// numOutChannels*OUTTILE_XSIZE elements apart -- ~2KB for a typical trunk
// width, so each fp16 store lands alone in its own 32-byte cache sector
// (~1/16 store efficiency). Moving `oc` onto the fast axis would fix the
// store but break the already-good load, a net loss. Instead each
// invocation now untransforms OC_VEC=4 *consecutive* channels (looped
// sequentially, reusing the same wTile[36] to avoid holding 4 tiles live at
// once) and issues a single f16vec4/vec4 store spanning all 4 -- since NHWC
// is channel-innermost, those 4 channels are contiguous in memory. That
// takes store efficiency from 1/16 to 1/4 (8 of 32 bytes useful per fp16
// sector) without touching the load side at all. OC_VEC falls back to 1
// (plain scalar store, byte-identical to the original shader) when
// numOutChannels isn't divisible by 4.

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
// Number of consecutive output channels each invocation untransforms and
// stores as one vector. 4 when numOutChannels%4==0 (the common case for
// trunk convolutions), else 1 (plain scalar path, see file header).
layout(constant_id = 10) const int OC_VEC = 1;

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
// Vector aliases of the output buffer for the OC_VEC==4 store path (and the
// ADD_TO_OUTPUT read-modify-write). Same descriptor-aliasing idiom as
// scale_bias_mask_act_nhwc.glsl's OutputBufV4/OutputBufV4H: the host binds
// the same VkBuffer at these binding numbers as at 1/3 (see bindingMap in
// vulkankernels.cpp), so this is purely a differently-typed view.
layout(binding = 4) buffer OutputBufV4 {
  vec4 data[];
}
outputBufV4;
layout(binding = 5) buffer OutputBufV4H {
  f16vec4 data[];
}
outputBufV4H;

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
vec4 loadOutput4(int idx4) {
  return USE_FP16_STORAGE == 1 ? vec4(outputBufV4H.data[idx4]) : outputBufV4.data[idx4];
}
void storeOutput4(int idx4, vec4 val) {
  if(ADD_TO_OUTPUT != 0)
    val += loadOutput4(idx4);
  if(USE_FP16_STORAGE == 1)
    outputBufV4H.data[idx4] = f16vec4(val);
  else
    outputBufV4.data[idx4] = val;
}

void main() {
  int tileX = int(gl_GlobalInvocationID.x);
  int tileY = int(gl_GlobalInvocationID.y);
  // With OC_VEC>1, the z axis walks batch*(numOutChannels/OC_VEC) channel
  // *groups* rather than individual channels -- ocBase is the first of the
  // OC_VEC consecutive channels this invocation is responsible for.
  int noc = int(gl_GlobalInvocationID.z);
  int numOcGroups = pc.numOutChannels / OC_VEC;
  int n = noc / numOcGroups;
  int ocBase = (noc % numOcGroups) * OC_VEC;

  if(tileX >= pc.numTilesX || tileY >= pc.numTilesY)
    return;

  int ntile = (n * pc.numTilesY + tileY) * pc.numTilesX + tileX;

  // Loop the OC_VEC channels sequentially, reusing wTile[36] for each one
  // rather than holding OC_VEC live input tiles at once (4x36 floats would
  // spill for F(4,3)). Only the small untransformed output tile survives
  // each iteration, collected into outVals for the vectorized store below.
  // 16 = OUTTILE_XSIZE*OUTTILE_YSIZE max (F(4,3)); 4 = max OC_VEC.
  float outVals[4][16];
  int transTileStride = pc.numOutChannelsPadded * pc.numTilesPadded;
  for(int j = 0; j < OC_VEC; j++) {
    int oc = ocBase + j;
    float wTile[36];

    int transBase = oc * pc.numTilesPadded + ntile;
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
      for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
        outVals[j][subY * OUTTILE_XSIZE + subX] = wTile[subY * INTILE_XSIZE + subX];
      }
    }
  }

  for(int subY = 0; subY < OUTTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY;
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX;
      if(y < pc.nnYLen && x < pc.nnXLen) {
        // NHWC: channel-innermost. The physical output tensor is stored with
        // the actual (unpadded) channel count as its stride, so the spatial
        // position selects a row of numOutChannels channels and ocBase
        // selects the first of the OC_VEC channels within that row. Those
        // OC_VEC channels are contiguous, so with numOutChannels%4==0 (the
        // OC_VEC==4 gate, enforced host-side) outIdx is divisible by 4 and
        // outIdx/4 is a valid vec4 element index.
        int outIdx = (n * pc.paddedSpatialSize + y * pc.nnXLen + x) * pc.numOutChannels + ocBase;
        int subIdx = subY * OUTTILE_XSIZE + subX;
        if(OC_VEC == 4) {
          storeOutput4(
            outIdx / 4, vec4(outVals[0][subIdx], outVals[1][subIdx], outVals[2][subIdx], outVals[3][subIdx]));
        } else {
          storeOutput(outIdx, outVals[0][subIdx]);
        }
      }
    }
  }
}

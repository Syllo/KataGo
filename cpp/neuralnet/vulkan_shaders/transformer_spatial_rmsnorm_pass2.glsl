// GLSL Compute Shader: Spatial RMSNorm Pass 2 - tree reduce partial sums
// Ports OpenCLKernels::transformerSpatialRMSNormReduce
// Dispatch: global(TILE_SIZE, batchSize, 1)  (single workgroup in X)

#version 450

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int TILE_SIZE = 64;

layout(local_size_x_id = 0) in;
layout(local_size_y = 1) in;
layout(local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numPartials;
  int tilesPerGroup;
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

shared float partials[TILE_SIZE];

void main() {
  int lid = int(gl_LocalInvocationID.x);
  int groupIdx = int(gl_WorkGroupID.x);
  int n = int(gl_GlobalInvocationID.y);

  int chunkStart = groupIdx * TILE_SIZE * pc.tilesPerGroup;
  float acc = 0.0;
  for(int t = 0; t < pc.tilesPerGroup; t++) {
    int idx = chunkStart + t * TILE_SIZE + lid;
    if(idx < pc.numPartials) {
      acc += inputBuf.data[n * pc.numPartials + idx];
    }
  }

  partials[lid] = acc;
  for(int span = TILE_SIZE / 2; span > 0; span /= 2) {
    barrier();
    if(lid < span)
      partials[lid] += partials[lid + span];
  }

  if(lid == 0) {
    // Single output per batch element (pass 2 always uses 1 workgroup in X)
    outputBuf.data[n] = partials[0];
  }
}

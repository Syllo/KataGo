// GLSL Compute Shader: Spatial RMSNorm Pass 2 - reduce partial sums (subgroup-shuffle variant)
// Ports OpenCLKernels::transformerSpatialRMSNormReduce
// Dispatch: global(TILE_SIZE, batchSize, 1)  (single workgroup in X)

#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require

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

shared float subgroupPartials[TILE_SIZE];

float subgroupSum(float value) {
  float sum = value;
  for(uint offset = gl_SubgroupSize / 2u; offset > 0u; offset >>= 1u) {
    sum += subgroupShuffleXor(sum, offset);
  }
  return sum;
}

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

  int lane = int(gl_SubgroupInvocationID);
  int subgroupID = int(gl_SubgroupID);
  int numSubgroups = int(gl_NumSubgroups);
  float localSubgroupSum = subgroupSum(acc);

  if(numSubgroups == 1) {
    if(lane == 0) {
      outputBuf.data[n] = localSubgroupSum;
    }
    return;
  }

  if(lane == 0) {
    subgroupPartials[subgroupID] = localSubgroupSum;
  }
  barrier();

  if(subgroupID == 0) {
    // Strided load so this is correct when numSubgroups > gl_SubgroupSize
    // (reachable when tileSize > subgroupSize -- e.g. tile=128 on Intel-class
    // subgroupSize=8). Each lane accumulates its own partials first, then the
    // intra-subgroup reduce finishes the sum. The host precondition
    // (requireFullSubgroups + fixed subgroup size, canUseRMSNormSubgroupVariant)
    // is also what makes stage 1's subgroupSum valid.
    float total = 0.0;
    for(uint i = uint(lane); i < uint(numSubgroups); i += gl_SubgroupSize)
      total += subgroupPartials[i];
    total = subgroupSum(total);
    if(lane == 0) {
      outputBuf.data[n] = total;
    }
  }
}

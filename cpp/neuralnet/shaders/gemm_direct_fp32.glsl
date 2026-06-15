// GLSL Compute Shader: Always-FP32 GEMM for MatMulLayer (FC heads)
// A: [M, K] (input: batchSize x numInChannels)
// B: [N, K] (weight, row-major transposed at upload)
// C: [M, N] (output)
// Dispatch: global(M, N, 1)

#version 450

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int LOCAL_SIZE_X = 8;
layout(constant_id = 1) const int LOCAL_SIZE_Y = 8;
layout(constant_id = 2) const int K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;  // batch elements
  int N;  // numOutChannels
}
pc;

layout(binding = 0) readonly buffer ABuf {
  float data[];
}
aBuf;
layout(binding = 1) readonly buffer BBuf {
  float data[];
}
bBuf;
layout(binding = 2) buffer CBuf {
  float data[];
}
cBuf;

void main() {
  int m = int(gl_GlobalInvocationID.x);
  int n = int(gl_GlobalInvocationID.y);
  if(m >= pc.M || n >= pc.N)
    return;

  float acc = 0.0;
  int aBase = m * K_SPEC;
  int bBase = n * K_SPEC;
  for(int k = 0; k < K_SPEC; k++) {
    acc += aBuf.data[aBase + k] * bBuf.data[bBase + k];
  }
  cBuf.data[m * pc.N + n] = acc;
}

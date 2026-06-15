// Probe shader for cooperative-matrix2 (VK_NV_cooperative_matrix2) capability
// testing. Compiles to a minimal SPIR-V module exercising the full feature set
// the backend's coopmat2 shaders rely on — tensor addressing, block loads,
// reductions, per-element operations, and matrix-type conversions. A successful
// compile gates KATAGO_HAS_COOPMAT2_SHADERS; which optional device feature bits
// the driver provides is queried at runtime via vkGetPhysicalDeviceFeatures2.
#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A {
  float16_t data_a[];
};
layout(binding = 1) readonly buffer B {
  float16_t data_b[];
};
layout(binding = 2) buffer C {
  float16_t data_c[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RopePair {
  f16vec2 pair;
};

// Row reductions, per-element operations, and block-load callbacks.
float maxReduce(const in float x, const in float y) {
  return max(x, y);
}
float sumReduce(const in float x, const in float y) {
  return x + y;
}
float scalePerElement(const in uint row, const in uint col, const in float x) {
  return x * 2.0;
}
float16_t blockLoad(const in RopePair raw, const in uint blockCoords[2], const in uint coordInBlock[2]) {
  const vec2 x = vec2(raw.pair);
  return coordInBlock[1] == 0u ? float16_t(x.x) : float16_t(x.y);
}

void main() {
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutA =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutB =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutC =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  layoutA = setTensorLayoutDimensionNV(layoutA, 16, 16);
  layoutB = setTensorLayoutDimensionNV(layoutB, 16, 16);
  layoutC = setTensorLayoutDimensionNV(layoutC, 16, 16);
  layoutA = setTensorLayoutStrideNV(layoutA, 1, 16);
  layoutB = setTensorLayoutStrideNV(layoutB, 1, 16);
  layoutC = setTensorLayoutStrideNV(layoutC, 16, 1);
  layoutA = setTensorLayoutClampValueNV(layoutA, 0u);
  layoutB = setTensorLayoutClampValueNV(layoutB, 0u);
  layoutC = setTensorLayoutClampValueNV(layoutC, 0u);
  tensorViewNV<2, false, 1, 0> transpose = createTensorViewNV(2, false, 1, 0);

  coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseA> a;
  coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseB> b;
  coopmat<float, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator> acc =
    coopmat<float, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  // Block load: per-element callback over a sliced tensor layout.
  coopMatLoadTensorNV(a, data_a, 0, sliceTensorLayoutNV(layoutA, 0, 16, 0, 16), blockLoad);
  // Tensor addressing: direct load with a transpose view.
  coopMatLoadTensorNV(b, data_b, 0, sliceTensorLayoutNV(layoutB, 0, 16, 0, 16), transpose);
  acc = coopMatMulAdd(a, b, acc);
  // Reductions: row reduction of the accumulator.
  coopmat<float, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator> tileMax;
  coopMatReduceNV(tileMax, acc, gl_CooperativeMatrixReduceRowNV, maxReduce);
  coopmat<float, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator> tileSum;
  coopMatReduceNV(tileSum, tileMax, gl_CooperativeMatrixReduceRowNV, sumReduce);
  // Per-element operations: applied to the reduced result.
  coopmat<float, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator> scaled;
  coopMatPerElementNV(scaled, tileSum, scalePerElement);
  // Conversions: FP32 accumulator converted to FP16 A/B operands.
  coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseA> a16 =
    coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseA>(scaled);
  coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseB> b16 =
    coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseB>(scaled);
  acc = coopMatMulAdd(a16, b16, acc);
  coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator> matC =
    coopmat<float16_t, gl_ScopeWorkgroup, 16, 16, gl_MatrixUseAccumulator>(acc);
  // Tensor addressing: store with a transpose view.
  coopMatStoreTensorNV(matC, data_c, 0, sliceTensorLayoutNV(layoutC, 0, 16, 0, 16), transpose);
}

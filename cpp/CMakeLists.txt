cmake_minimum_required(VERSION 3.18.2)
if(USE_BACKEND STREQUAL "METAL")
  project(katago LANGUAGES CXX Swift)
else()
  project(katago)
endif()

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES)
  # Default to RelWithDebInfo if no build type specified
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

include_directories(external)
include_directories(SYSTEM external/tclap-1.2.5/include) #SYSTEM suppresses warnings from external library
include_directories(SYSTEM external/filesystem-1.5.8/include) #SYSTEM suppresses a few warnings

#--------------------------- PLATFORM SPECIFIC -------------------------------------------------------------------------

if(APPLE)
  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  if(IS_DIRECTORY /usr/local/include)
    include_directories(/usr/local/include)
  endif()
  if(IS_DIRECTORY /usr/local/lib)
    link_directories(/usr/local/lib)
  endif()
endif()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset   "${Esc}[m")
  set(ColorBold    "${Esc}[1m")
  set(ColorRed     "${Esc}[31m")
  set(ColorBoldRed "${ColorRed}${ColorBold}")
endif()

#--------------------------- CMAKE VARIABLES (partly for Cmake GUI) ----------------------------------------------------

set(BUILD_DISTRIBUTED 0 CACHE BOOL "Build with http support for contributing to distributed training")
set(USE_BACKEND CACHE STRING "Neural net backend")
string(TOUPPER "${USE_BACKEND}" USE_BACKEND)
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN METAL VULKAN)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 50. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")

set(USE_CACHE_TENSORRT_PLAN 0 CACHE BOOL "Use TENSORRT plan cache. May use a lot of disk space. Only applies when USE_BACKEND is TENSORRT.")
mark_as_advanced(USE_CACHE_TENSORRT_PLAN)

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message(STATUS "Building 'katago' executable for GTP engine and other tools.")
if(USE_BACKEND STREQUAL "CUDA")
  message(STATUS "-DUSE_BACKEND=CUDA, using CUDA backend.")

  enable_language(CUDA)

  set(CUDA_STANDARD 11)
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/cudabackend.cpp
    neuralnet/cudautils.cpp
    neuralnet/cudahelpers.cu
  )
  # The vendored header-only cudnn_frontend library trips GCC's -Wnull-dereference from its -O2
  # interprocedural analysis (no source location, so an in-file #pragma can't suppress it). It's
  # benign third-party code; silence it just for the one source that includes cudnn_frontend.h.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(neuralnet/cudabackend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-null-dereference")
  endif()
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 88 89 90 100 110 103 120 121)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90 120)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
    set(CMAKE_CUDA_ARCHITECTURES 30 35 37 50 52 53 60 61 62 70 72 75)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  else()
    message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
    set(CMAKE_CUDA_ARCHITECTURES 30 37 53 70)
  endif()
elseif(USE_BACKEND STREQUAL "TENSORRT")
  message(STATUS "-DUSE_BACKEND=TENSORRT, using TensorRT backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/trtbackend.cpp
    )
  if(USE_CACHE_TENSORRT_PLAN AND (NOT BUILD_DISTRIBUTED))
    message(STATUS "-DUSE_CACHE_TENSORRT_PLAN is set, using TENSORRT plan cache.")
    add_compile_definitions(CACHE_TENSORRT_PLAN)
  elseif(USE_CACHE_TENSORRT_PLAN AND BUILD_DISTRIBUTED)
    message(FATAL_ERROR "Combining USE_CACHE_TENSORRT_PLAN with BUILD_DISTRIBUTED is not supported - it would consume excessive disk space and might worsen performance every time models are updated. Use only one at a time in a given build of KataGo.")
  endif()
elseif(USE_BACKEND STREQUAL "METAL")
  message(STATUS "-DUSE_BACKEND=METAL, using Metal backend with hybrid MPSGraph + CoreML execution.")
  if(NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
    message(FATAL_ERROR "Bidirectional C++ Interop requires Ninja generator. Have ${CMAKE_GENERATOR}")
  endif()
  if("${CMAKE_Swift_COMPILER_VERSION}" VERSION_LESS 5.9)
    message(FATAL_ERROR "Bidirectional C++ Interop requires Swift 5.9 or greater. Have ${CMAKE_Swift_COMPILER_VERSION}")
  endif()
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message(FATAL_ERROR "Project requires building with AppleClang. Have ${CMAKE_CXX_COMPILER_ID}")
  endif()
  add_subdirectory(external/katagocoreml)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/macos/cmake/modules")

  if (NOT CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE CMAKE_OSX_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

  include(InitializeSwift)
  include(AddSwift)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/metalbackend.cpp
    )
  add_library(KataGoSwift STATIC
    neuralnet/metalbackend.swift
    neuralnet/metallayers.swift)
  _swift_generate_cxx_header(
    KataGoSwift
    "${CMAKE_CURRENT_BINARY_DIR}/include/KataGoSwift/KataGoSwift-swift.h")
  target_include_directories(KataGoSwift PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
  set_target_properties(KataGoSwift PROPERTIES Swift_MODULE_NAME "KataGoSwift")
  target_compile_options(KataGoSwift PUBLIC
    "$<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default>")
elseif(USE_BACKEND STREQUAL "OPENCL")
  message(STATUS "-DUSE_BACKEND=OPENCL, using OpenCL backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/openclbackend.cpp
    neuralnet/openclkernels.cpp
    neuralnet/openclhelpers.cpp
    neuralnet/opencltuner.cpp
    )
elseif(USE_BACKEND STREQUAL "EIGEN")
  message(STATUS "-DUSE_BACKEND=EIGEN, using Eigen CPU backend.")
  if(NOT USE_AVX2)
    message(STATUS "You can also specify USE_AVX2 (-DUSE_AVX2=1 on command line) if you have a modern CPU for better performance.")
  endif()
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/eigenbackend.cpp
    )
elseif(USE_BACKEND STREQUAL "VULKAN")
  message(STATUS "-DUSE_BACKEND=VULKAN, using Vulkan Compute backend.")

  find_package(Vulkan REQUIRED)
  find_program(GLSLC_EXECUTABLE glslc
    HINTS
      ${Vulkan_GLSLC_EXECUTABLE}
      "$ENV{VULKAN_SDK}/bin"
      "$ENV{VK_SDK_PATH}/bin"
  )
  if(NOT GLSLC_EXECUTABLE)
    # Fall back to glslangValidator (from the glslang-tools package)
    find_program(GLSLC_EXECUTABLE glslangValidator)
    if(GLSLC_EXECUTABLE)
      # glslangValidator uses different flags: -V for SPIR-V, -o for output, --stdin for stage
      set(GLSLC_FLAGS -V --target-env vulkan1.1 -S comp)
      message(STATUS "glslc not found, using glslangValidator: ${GLSLC_EXECUTABLE}")
    else()
      message(FATAL_ERROR "Neither glslc nor glslangValidator found. Install 'glslc' (shaderc) or 'glslang-tools'.")
    endif()
  else()
    set(GLSLC_FLAGS -fshader-stage=compute -O --target-env=vulkan1.1)
    message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")
  endif()

  set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/vulkan_shaders)
  set(SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${SHADER_GEN_DIR})

  # Probe whether glslc accepts -O for a shader family; cache -O or -O0. No-op
  # (leaves CACHE_VAR unset) for glslangValidator, which has no -O — the compile
  # helper's glslangValidator branch ignores the flag anyway. Include the source
  # hash in the cache key: optimizer support can change when a shader is edited,
  # so an old -O0 fallback must not silently survive a source improvement.
  function(katago_probe_glslc_opt_level CACHE_VAR PROBE_SRC TARGET_ENV DESC)
    if(NOT GLSLC_EXECUTABLE MATCHES "glslc")
      return()
    endif()
    file(SHA256 ${PROBE_SRC} _source_hash)
    set(_hash_var "${CACHE_VAR}_SOURCE_HASH")
    if(DEFINED ${CACHE_VAR} AND DEFINED ${_hash_var} AND
       "${${_hash_var}}" STREQUAL "${_source_hash}")
      return()
    endif()
    set(_probe_out ${CMAKE_BINARY_DIR}/CMakeFiles/${CACHE_VAR}_probe.spv)
    set(_probe_defines ${ARGN})
    execute_process(
      COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute -O ${_probe_defines} --target-env=${TARGET_ENV}
              ${PROBE_SRC} -o ${_probe_out}
      RESULT_VARIABLE _res OUTPUT_QUIET ERROR_QUIET)
    file(REMOVE ${_probe_out})
    if(_res EQUAL 0)
      set(${CACHE_VAR} "-O" CACHE INTERNAL "glslc optimization flag for ${DESC} shaders")
      message(STATUS "glslc accepts -O for ${DESC} shaders; they will be optimized")
    else()
      set(${CACHE_VAR} "-O0" CACHE INTERNAL "glslc optimization flag for ${DESC} shaders")
      message(STATUS "glslc rejects -O for ${DESC} shaders; compiling with -O0")
    endif()
    set(${_hash_var} "${_source_hash}" CACHE INTERNAL "source hash probed for ${DESC} shaders")
  endfunction()

  # Probe whether the shader compiler can emit SPV_KHR_float_controls execution
  # mode RoundingModeRTE for 16-bit floats. If yes, we build a second SPIR-V
  # variant for each shader and select it at runtime only on devices whose
  # shaderRoundingModeRTEFloat16 property permits it.
  function(katago_probe_rte_f16_shader_support)
    if(DEFINED KATAGO_HAS_RTE_F16_SHADERS)
      return()
    endif()
    set(_rte_probe_src ${CMAKE_BINARY_DIR}/CMakeFiles/rte_f16_probe.comp)
    set(_rte_probe_out ${CMAKE_BINARY_DIR}/CMakeFiles/rte_f16_probe.spv)
    file(WRITE ${_rte_probe_src}
      "#version 450\n"
      "#extension GL_EXT_spirv_intrinsics : require\n"
      "#ifdef KATAGO_VULKAN_RTE_F16\n"
      "spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [4467], 4462, 16);\n"
      "#endif\n"
      "layout(local_size_x = 1) in;\n"
      "void main() {}\n"
    )
    if(GLSLC_EXECUTABLE MATCHES "glslc")
      set(_rte_probe_command
        ${GLSLC_EXECUTABLE} -fshader-stage=compute -O0 -DKATAGO_VULKAN_RTE_F16=1 --target-env=vulkan1.1
        ${_rte_probe_src} -o ${_rte_probe_out})
    else()
      set(_rte_probe_command
        ${GLSLC_EXECUTABLE} -V --target-env vulkan1.1 -S comp -DKATAGO_VULKAN_RTE_F16=1
        ${_rte_probe_src} -o ${_rte_probe_out})
    endif()
    execute_process(
      COMMAND ${_rte_probe_command}
      RESULT_VARIABLE _rte_probe_result
      OUTPUT_VARIABLE _rte_probe_stdout
      ERROR_VARIABLE _rte_probe_stderr
    )
    set(_rte_probe_verified TRUE)
    if(_rte_probe_result EQUAL 0)
      find_program(SPIRV_DIS_EXECUTABLE spirv-dis HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VK_SDK_PATH}/bin")
      if(SPIRV_DIS_EXECUTABLE)
        execute_process(
          COMMAND ${SPIRV_DIS_EXECUTABLE} ${_rte_probe_out}
          RESULT_VARIABLE _rte_probe_dis_result
          OUTPUT_VARIABLE _rte_probe_dis_output
          ERROR_QUIET
        )
        if(NOT _rte_probe_dis_result EQUAL 0)
          set(_rte_probe_verified FALSE)
        endif()
        string(FIND "${_rte_probe_dis_output}" "OpExtension \"SPV_KHR_float_controls\"" _rte_probe_ext_pos)
        string(FIND "${_rte_probe_dis_output}" "OpExecutionMode %main RoundingModeRTE 16" _rte_probe_mode_pos)
        if(_rte_probe_ext_pos EQUAL -1 OR _rte_probe_mode_pos EQUAL -1)
          set(_rte_probe_verified FALSE)
        endif()
      endif()
    endif()
    file(REMOVE ${_rte_probe_src} ${_rte_probe_out})
    if(_rte_probe_result EQUAL 0 AND _rte_probe_verified)
      set(KATAGO_HAS_RTE_F16_SHADERS ON CACHE INTERNAL "shader toolchain supports SPV_KHR_float_controls RTE f16")
      message(STATUS "Shader toolchain supports SPV_KHR_float_controls RoundingModeRTE f16; building RTE shader variants")
    else()
      set(KATAGO_HAS_RTE_F16_SHADERS OFF CACHE INTERNAL "shader toolchain supports SPV_KHR_float_controls RTE f16")
      message(STATUS "${ColorRed}Shader toolchain cannot emit SPV_KHR_float_controls RoundingModeRTE f16; RTE shader variants disabled.${ColorReset}")
      message(STATUS "RTE f16 probe command: ${_rte_probe_command}")
      if(NOT "${_rte_probe_stdout}" STREQUAL "")
        message(STATUS "RTE f16 probe stdout:\n${_rte_probe_stdout}")
      endif()
      if(NOT "${_rte_probe_stderr}" STREQUAL "")
        message(STATUS "RTE f16 probe stderr:\n${_rte_probe_stderr}")
      endif()
    endif()
  endfunction()

  # Probe whether the shader toolchain can compile a representative shader for a
  # shader capability family (coopmat1/coopmat2), caching the result ON/OFF in
  # CACHE_VAR so a configure re-run does not re-probe. On failure the offending
  # command and its stdout/stderr are printed to help diagnose toolchain gaps.
  function(katago_probe_shader_family CACHE_VAR FAMILY CAPABILITY PROBE_SRC TARGET_ENV)
    if(DEFINED ${CACHE_VAR})
      return()
    endif()
    set(_probe_out ${CMAKE_BINARY_DIR}/CMakeFiles/${FAMILY}_probe.spv)
    if(GLSLC_EXECUTABLE MATCHES "glslc")
      set(_probe_command
        ${GLSLC_EXECUTABLE} -fshader-stage=compute -O0 --target-env=${TARGET_ENV} ${PROBE_SRC} -o ${_probe_out})
    else()
      set(_probe_command
        ${GLSLC_EXECUTABLE} -V --target-env ${TARGET_ENV} -S comp ${PROBE_SRC} -o ${_probe_out})
    endif()
    execute_process(
      COMMAND ${_probe_command}
      RESULT_VARIABLE _probe_result
      OUTPUT_VARIABLE _probe_stdout
      ERROR_VARIABLE _probe_stderr)
    file(REMOVE ${_probe_out})
    if(_probe_result EQUAL 0)
      set(${CACHE_VAR} ON CACHE INTERNAL "glslc/glslang supports ${CAPABILITY}")
      message(STATUS "Shader toolchain supports ${CAPABILITY}; building ${FAMILY} shaders")
    else()
      set(${CACHE_VAR} OFF CACHE INTERNAL "glslc/glslang supports ${CAPABILITY}")
      message(STATUS "${ColorRed}Shader toolchain lacks ${CAPABILITY} support; ${FAMILY} path disabled (backend falls back). Upgrade shaderc/glslang to enable it.${ColorReset}")
      message(STATUS "${FAMILY} probe command: ${_probe_command}")
      if(NOT "${_probe_stdout}" STREQUAL "")
        message(STATUS "${FAMILY} probe stdout:\n${_probe_stdout}")
      endif()
      if(NOT "${_probe_stderr}" STREQUAL "")
        message(STATUS "${FAMILY} probe stderr:\n${_probe_stderr}")
      endif()
    endif()
  endfunction()


  # add_custom_command per shader in ARGN; append each .spv to the list named by
  # OUT_VAR in the caller (directory) scope. Branches glslc vs glslangValidator;
  # OPT_LEVEL/TARGET_ENV are per-family. OPT_LEVEL is ignored on the glslangValidator
  # path (empty is fine there).
  function(katago_compile_shader_group OUT_VAR OPT_LEVEL TARGET_ENV DESC)
    set(_outputs ${${OUT_VAR}})
    foreach(GLSL_FILE ${ARGN})
      string(REPLACE ".glsl" ".spv" SPV_FILE ${GLSL_FILE})
      katago_compile_shader(_outputs ${OPT_LEVEL} ${TARGET_ENV} "${DESC}" ${GLSL_FILE} ${SPV_FILE})
    endforeach()
    set(${OUT_VAR} ${_outputs} PARENT_SCOPE)
  endfunction()

  # Compile an additional named SPIR-V variant from an existing GLSL source.
  # This is used for accumulator-type variants whose source is otherwise
  # identical, keeping the native convolution implementations in lockstep.
  function(katago_compile_shader_variant OUT_VAR OPT_LEVEL TARGET_ENV DESC SOURCE_FILE OUTPUT_FILE)
    set(_variant_outputs ${${OUT_VAR}})
    katago_compile_shader(_variant_outputs ${OPT_LEVEL} ${TARGET_ENV} "${DESC}" ${SOURCE_FILE} ${OUTPUT_FILE} ${ARGN})
    set(${OUT_VAR} ${_variant_outputs} PARENT_SCOPE)
  endfunction()

  # Single-shader core shared by the group and variant helpers. SOURCE_FILE is
  # the .glsl under SHADER_SRC_DIR, OUTPUT_FILE the .spv name under SHADER_GEN_DIR
  # (the group derives it from the source name; the variant passes it explicitly).
  # Remaining ARGN entries are -D defines applied to both the normal and RTE compile.
  function(katago_compile_shader OUT_VAR OPT_LEVEL TARGET_ENV DESC SOURCE_FILE OUTPUT_FILE)
    set(_outputs ${${OUT_VAR}})
    set(SPV_PATH ${SHADER_GEN_DIR}/${OUTPUT_FILE})
    set(_defines "")
    foreach(_def ${ARGN})
      list(APPEND _defines -D${_def})
    endforeach()
    if(GLSLC_EXECUTABLE MATCHES "glslc")
      add_custom_command(OUTPUT ${SPV_PATH}
        COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute ${OPT_LEVEL} ${_defines}
                --target-env=${TARGET_ENV} ${SHADER_SRC_DIR}/${SOURCE_FILE} -o ${SPV_PATH}
        DEPENDS ${SHADER_SRC_DIR}/${SOURCE_FILE}
        COMMENT "Compiling ${DESC}shader ${SOURCE_FILE} to SPIR-V (${OPT_LEVEL})" VERBATIM)
    else()
      add_custom_command(OUTPUT ${SPV_PATH}
        COMMAND ${GLSLC_EXECUTABLE} -V --target-env ${TARGET_ENV} -S comp ${_defines}
                ${SHADER_SRC_DIR}/${SOURCE_FILE} -o ${SPV_PATH}
        DEPENDS ${SHADER_SRC_DIR}/${SOURCE_FILE}
        COMMENT "Compiling ${DESC}shader ${SOURCE_FILE} to SPIR-V" VERBATIM)
    endif()
    list(APPEND _outputs ${SPV_PATH})
    if(KATAGO_HAS_RTE_F16_SHADERS)
      string(REPLACE ".spv" "_rte.spv" RTE_OUTPUT_FILE ${OUTPUT_FILE})
      set(RTE_SPV_PATH ${SHADER_GEN_DIR}/${RTE_OUTPUT_FILE})
      if(GLSLC_EXECUTABLE MATCHES "glslc")
        add_custom_command(OUTPUT ${RTE_SPV_PATH}
          COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute ${OPT_LEVEL} ${_defines}
                  -DKATAGO_VULKAN_RTE_F16=1 --target-env=${TARGET_ENV}
                  ${SHADER_SRC_DIR}/${SOURCE_FILE} -o ${RTE_SPV_PATH}
          DEPENDS ${SHADER_SRC_DIR}/${SOURCE_FILE}
          COMMENT "Compiling ${DESC}RTE shader ${RTE_OUTPUT_FILE} to SPIR-V (${OPT_LEVEL})" VERBATIM)
      else()
        add_custom_command(OUTPUT ${RTE_SPV_PATH}
          COMMAND ${GLSLC_EXECUTABLE} -V --target-env ${TARGET_ENV} -S comp ${_defines}
                  -DKATAGO_VULKAN_RTE_F16=1 ${SHADER_SRC_DIR}/${SOURCE_FILE} -o ${RTE_SPV_PATH}
          DEPENDS ${SHADER_SRC_DIR}/${SOURCE_FILE}
          COMMENT "Compiling ${DESC}RTE shader ${RTE_OUTPUT_FILE} to SPIR-V" VERBATIM)
      endif()
      list(APPEND _outputs ${RTE_SPV_PATH})
    endif()
    set(${OUT_VAR} ${_outputs} PARENT_SCOPE)
  endfunction()

  katago_probe_rte_f16_shader_support()

  set(GLSL_SHADER_FILES
    winograd_transform_nhwc.glsl
    winograd_untransform_nhwc.glsl
    conv2d_direct_nhwc.glsl
    scale_bias_mask_act_nhwc.glsl
    scale_bias_mask_act_inplace_nhwc.glsl
    nchw_to_nhwc_smallchannel.glsl
    nchw_to_nhwc_tiled.glsl
    winograd_gemm_tiled.glsl
    gemm_strided_tiled_nhwc.glsl
    gemm_direct_fp32.glsl
    add_channel_biases_nhwc.glsl
    add_cbias_act_nc.glsl
    add_pointwise.glsl
    extract_channel0_nhwc.glsl
    sum_mask_spatial.glsl
    gpool_mask_nhwc.glsl
    value_head_pool_nhwc.glsl
    transformer_rmsnorm_nhwc.glsl
    transformer_rmsnorm_nhwc_subgroup.glsl
    transformer_spatial_rmsnorm_pass2.glsl
    transformer_spatial_rmsnorm_pass2_subgroup.glsl
    transformer_spatial_rmsnorm_nhwc_pass1.glsl
    transformer_spatial_rmsnorm_nhwc_pass3.glsl
    transformer_attention_tiled_nhwc.glsl
    transformer_swiglu.glsl
  )

  set(SPIRV_OUTPUTS "")
  katago_compile_shader_group(SPIRV_OUTPUTS "-O" vulkan1.1 "GLSL " ${GLSL_SHADER_FILES})

  # DOT2 shaders use SPV_VALVE_mixed_float_dot_product (capability ids 6912/6913).
  # Older SPIRV-Tools rejects this as an unknown capability during optimization.
  # Probe glslc at configure time: try compiling an AccF16 DOT2 shader with -O, and
  # cache whether that toolchain accepts it. Then hard-code the resulting flag
  # into the per-shader add_custom_command (portable: no sh -c fallback).
  # After a glslc/SPIRV-Tools upgrade, remove CMakeCache.txt or configure with
  # -U KATAGO_GLSLC_DOT2_MIXED_OPT_LEVEL to force this cached probe to run again.
  set(GLSL_DOT2_SHADER_FILES
    winograd_gemm_dot2_accf32.glsl
    transformer_attention_dot2_accf32_nhwc.glsl
    winograd_gemm_dot2_accf16.glsl
    gemm_strided_dot2_accf32_nhwc.glsl
    gemm_strided_dot2_accf16_nhwc.glsl
  )
  katago_probe_glslc_opt_level(KATAGO_GLSLC_DOT2_MIXED_OPT_LEVEL
    ${SHADER_SRC_DIR}/gemm_strided_dot2_accf16_nhwc.glsl vulkan1.1 "DOT2")
  katago_compile_shader_group(SPIRV_OUTPUTS ${KATAGO_GLSLC_DOT2_MIXED_OPT_LEVEL}
    vulkan1.1 "DOT2 GLSL " ${GLSL_DOT2_SHADER_FILES})

  # Cooperative-matrix (coopmat1, VK_KHR_cooperative_matrix) GEMM shaders.
  # Compiled with --target-env=vulkan1.1 (which still emits SPIR-V 1.3, the
  # minimum the coopmat types require) for the widest driver compatibility — a
  # vulkan1.3 target would emit SPIR-V 1.6 and needlessly narrow it.
  set(GLSL_COOPMAT_SHADER_FILES
    winograd_gemm_coopmat1_accf32.glsl
    winograd_gemm_coopmat1_accf16.glsl
    gemm_strided_coopmat1_accf16_nhwc.glsl
    conv_implicit_gemm_coopmat1_accf32_nhwc.glsl
    gemm_strided_coopmat1_accf32_nhwc.glsl
    transformer_attention_coopmat1_nhwc.glsl
    transformer_attention_coopmat1_split_k_nhwc.glsl
    transformer_attention_split_k_resolve_nhwc.glsl
  )
  # Probe the toolchain for GL_KHR_cooperative_matrix support at configure time.
  # Old shaderc/glslang lack it; rather than fail the whole Vulkan build for
  # users who have no coopmat hardware anyway, skip the coopmat shaders and let
  # the backend fall back (guarded by KATAGO_VULKAN_HAS_COOPMAT_SHADERS in C++).
  # Force a re-probe after a toolchain upgrade with -U KATAGO_HAS_COOPMAT_SHADERS.
  if(NOT DEFINED KATAGO_HAS_COOPMAT_SHADERS)
    katago_probe_shader_family(KATAGO_HAS_COOPMAT_SHADERS "coopmat GEMM" "GL_KHR_cooperative_matrix"
      ${SHADER_SRC_DIR}/winograd_gemm_coopmat1_accf32.glsl vulkan1.1)
  endif()
  if(KATAGO_HAS_COOPMAT_SHADERS)
    katago_probe_glslc_opt_level(KATAGO_GLSLC_COOPMAT_OPT_LEVEL
      ${SHADER_SRC_DIR}/winograd_gemm_coopmat1_accf32.glsl vulkan1.1 "coopmat GLSL")
    katago_compile_shader_group(SPIRV_OUTPUTS ${KATAGO_GLSLC_COOPMAT_OPT_LEVEL}
      vulkan1.1 "coopmat GLSL " ${GLSL_COOPMAT_SHADER_FILES})
    # maintenance1 is optional. Its register-softmax path lives in distinct
    # shaders rather than being preprocessor-selected inside the portable pair.
    if(NOT DEFINED KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS)
      set(_coopmat_maint1_probe_out ${CMAKE_BINARY_DIR}/CMakeFiles/coopmat_maint1_probe.spv)
      execute_process(
        COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute -O0 --target-env=vulkan1.1
                ${SHADER_SRC_DIR}/transformer_attention_coopmat1_maintenance1_nhwc.glsl -o ${_coopmat_maint1_probe_out}
        RESULT_VARIABLE _coopmat_maint1_probe_result OUTPUT_QUIET ERROR_QUIET)
      file(REMOVE ${_coopmat_maint1_probe_out})
      set(KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS ${_coopmat_maint1_probe_result} CACHE INTERNAL
        "glslc cooperative matrix maintenance1 probe result")
      if(_coopmat_maint1_probe_result EQUAL 0)
        set(KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS ON CACHE INTERNAL "" FORCE)
      else()
        set(KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS OFF CACHE INTERNAL "" FORCE)
      endif()
    endif()
    if(KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS)
      katago_probe_glslc_opt_level(KATAGO_GLSLC_COOPMAT_MAINT1_OPT_LEVEL
        ${SHADER_SRC_DIR}/transformer_attention_coopmat1_maintenance1_nhwc.glsl vulkan1.1 "coopmat maintenance1")
      katago_compile_shader_group(SPIRV_OUTPUTS ${KATAGO_GLSLC_COOPMAT_MAINT1_OPT_LEVEL}
        vulkan1.1 "coopmat maintenance1 " transformer_attention_coopmat1_maintenance1_nhwc.glsl transformer_attention_coopmat1_maintenance1_split_k_nhwc.glsl)
    endif()
    katago_compile_shader_variant(
      SPIRV_OUTPUTS ${KATAGO_GLSLC_COOPMAT_OPT_LEVEL} vulkan1.1 "coopmat AccF16 "
      conv_implicit_gemm_coopmat1_accf32_nhwc.glsl
      conv_implicit_gemm_coopmat1_accf16_nhwc.spv
      KATAGO_VULKAN_ACCUMULATE_FP16=1)
  endif()

  # Cooperative-matrix2 (VK_NV_cooperative_matrix2) GEMM shaders. These require
  # SPV_NV_cooperative_matrix2 / SPV_NV_tensor_addressing, which require SPIR-V
  # 1.6, so target Vulkan 1.3.
  set(GLSL_COOPMAT2_SHADER_FILES
    winograd_gemm_coopmat2_accf32.glsl
    winograd_gemm_coopmat2_accf16.glsl
    gemm_strided_coopmat2_accf16_nhwc.glsl
    conv_implicit_gemm_coopmat2_accf32_nhwc.glsl
    gemm_strided_coopmat2_accf32_nhwc.glsl
    transformer_attention_coopmat2_accf32_nhwc.glsl
  )
  # The probe shader exercises every coopmat2 op the backend's shaders use (tensor
  # addressing, block loads, reductions, per-element operations, and conversions),
  # so a successful compile confirms the toolchain can build them all. Which
  # optional device feature bits the driver provides is queried at runtime via
  # vkGetPhysicalDeviceFeatures2, not here.
  if(NOT DEFINED KATAGO_HAS_COOPMAT2_SHADERS)
    katago_probe_shader_family(KATAGO_HAS_COOPMAT2_SHADERS "coopmat2 GEMM" "GL_NV_cooperative_matrix2"
      ${SHADER_SRC_DIR}/coopmat2_probe.glsl vulkan1.3)
  endif()
  # Compile coopmat2 shaders only when supported.
  if(KATAGO_HAS_COOPMAT2_SHADERS)
    katago_probe_glslc_opt_level(KATAGO_GLSLC_COOPMAT2_OPT_LEVEL
      ${SHADER_SRC_DIR}/gemm_strided_coopmat2_accf32_nhwc.glsl vulkan1.3 "coopmat2 GLSL")
    katago_compile_shader_group(SPIRV_OUTPUTS ${KATAGO_GLSLC_COOPMAT2_OPT_LEVEL}
      vulkan1.3 "coopmat2 GLSL " ${GLSL_COOPMAT2_SHADER_FILES})
    katago_compile_shader_variant(
      SPIRV_OUTPUTS ${KATAGO_GLSLC_COOPMAT2_OPT_LEVEL} vulkan1.3 "coopmat2 AccF16 "
      conv_implicit_gemm_coopmat2_accf32_nhwc.glsl
      conv_implicit_gemm_coopmat2_accf16_nhwc.spv
      KATAGO_VULKAN_ACCUMULATE_FP16=1)
  endif()


  set(GENERATED_CPP ${SHADER_GEN_DIR}/vulkanshaders_generated.cpp)
  set(GENERATED_H   ${SHADER_GEN_DIR}/vulkanshaders_generated.h)
  set(SPIRV_MANIFEST ${SHADER_GEN_DIR}/vulkanshaders_spv_files.txt)

  string(REPLACE ";" "\n" SPIRV_MANIFEST_CONTENT "${SPIRV_OUTPUTS}")
  file(WRITE ${SPIRV_MANIFEST} "${SPIRV_MANIFEST_CONTENT}\n")

  add_custom_command(
    OUTPUT ${GENERATED_CPP} ${GENERATED_H}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_MANIFEST=${SPIRV_MANIFEST}
            -DOUTPUT_CPP=${GENERATED_CPP}
            -DOUTPUT_H=${GENERATED_H}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${SPIRV_OUTPUTS} ${SPIRV_MANIFEST} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    COMMENT "Embedding SPIR-V shaders into C++ arrays"
    VERBATIM
  )

  set(NEURALNET_BACKEND_SOURCES
    neuralnet/vulkanbackend.cpp
    neuralnet/vulkanhelpers.cpp
    neuralnet/vulkankernels.cpp
    neuralnet/vulkanlayers.cpp
    neuralnet/vulkantunebench.cpp
    neuralnet/vulkantuneparams.cpp
    neuralnet/vulkantuner.cpp
    ${GENERATED_CPP}
  )
  include_directories(${SHADER_GEN_DIR})
elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN to compile with the respective backend.${ColorReset}")
  set(NEURALNET_BACKEND_SOURCES neuralnet/dummybackend.cpp)
else()
  message(FATAL_ERROR "Unrecognized backend: " ${USE_BACKEND})
endif()


#--------------------------- TCMALLOC ----------------------------------------------------------------------------------

if(USE_TCMALLOC)
  message(STATUS "-DUSE_TCMALLOC=1 is set, using tcmalloc as the allocator")
  find_library(TCMALLOC_LIB NAMES tcmalloc_minimal HINTS /usr)
  if(NOT TCMALLOC_LIB)
    message(FATAL_ERROR "Could not find tcmalloc")
  endif()
endif()

# set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
# find_package(Gperftools REQUIRED)

#--------------------------- GIT ---------------------------------------------------------------------------------------

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  message(STATUS "-DNO_GIT_REVISION=1 is set, avoiding including the Git revision in compiled executable")
  unset(GIT_HEADER_FILE_ALWAYS_UPDATED)
else()
  if(NO_GIT_REVISION AND BUILD_DISTRIBUTED)
    message(STATUS "${ColorRed}NO_GIT_REVISION is set, but BUILD_DISTRIBUTED is also set and distributed requires git revision, so ignoring NO_GIT_REVISION.${ColorReset}")
  elseif(BUILD_DISTRIBUTED)
    message(STATUS "Including Git revision in the compiled executable")
  else()
    message(STATUS "Including Git revision in the compiled executable, specify -DNO_GIT_REVISION=1 to disable")
  endif()
  find_package(Git)
  if(NOT GIT_FOUND)
    set(GIT_EXECUTABLE ${GIT_EXECUTABLE} CACHE FILEPATH "Path to git executable")
    mark_as_advanced(CLEAR GIT_EXECUTABLE)
    if(BUILD_DISTRIBUTED)
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found, specify GIT_EXECUTABLE as the path to the git executable.${ColorReset}")
    else()
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found. Either specify GIT_EXECUTABLE as the path to the git executable, or use NO_GIT_REVISION to disable.${ColorReset}")
    endif()
  endif()
  set(GIT_HEADER_FILE_TEMPLATE_BARE program/gitinfotemplate.h)
  set(GIT_HEADER_FILE_ALWAYS_UPDATED_BARE program/gitinfoupdated.h)
  set(GIT_HEADER_FILE_BARE program/gitinfo.h)
  set(GIT_HEADER_FILE_TEMPLATE ${CMAKE_SOURCE_DIR}/${GIT_HEADER_FILE_TEMPLATE_BARE})
  set(GIT_HEADER_FILE_ALWAYS_UPDATED ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_ALWAYS_UPDATED_BARE})
  set(GIT_HEADER_FILE ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_BARE})
  # Embeds the current git SHA into the binary on every build, while only
  # triggering recompilation of dependents when the SHA actually changes.
  #
  # How it works:
  #   - ALWAYS_UPDATED is the rule's OUTPUT and is removed at the end of the
  #     commands, so it is always missing at the start of the next build,
  #     which forces this rule to re-fire.
  #   - copy_if_different updates GIT_HEADER_FILE only when the new SHA
  #     differs from the previous one, leaving its mtime untouched otherwise.
  #
  # Both files must be listed as OUTPUTs for Ninja. Ninja's restat (set
  # automatically for custom_commands) re-checks every declared OUTPUT's
  # mtime after the rule runs, and only then propagates rebuilds. Listing
  # GIT_HEADER_FILE is what lets ninja notice an actual SHA change and
  # rebuild dependents such as main.cpp; without it, dependents stay stale
  # across commits in incremental builds. The Make generator is more
  # pessimistic about generated headers and doesn't need this, and adding
  # it causes unnecessary re-linking on every build.
  set(GIT_CUSTOM_COMMAND_OUTPUTS ${GIT_HEADER_FILE_ALWAYS_UPDATED})
  if(CMAKE_GENERATOR MATCHES "Ninja")
    list(APPEND GIT_CUSTOM_COMMAND_OUTPUTS ${GIT_HEADER_FILE})
  endif()
  add_custom_command(
    OUTPUT ${GIT_CUSTOM_COMMAND_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E copy ${GIT_HEADER_FILE_TEMPLATE} ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${GIT_EXECUTABLE} describe --match=DummyTagNotExisting --always --abbrev=40 --dirty >> ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GIT_HEADER_FILE_ALWAYS_UPDATED} ${GIT_HEADER_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
    )
endif()

#--------------------------- KATAGO COMPILING AND LINKING --------------------------------------------------------------

add_executable(katago
  core/global.cpp
  core/base64.cpp
  core/bsearch.cpp
  core/commandloop.cpp
  core/config_parser.cpp
  core/datetime.cpp
  core/elo.cpp
  core/fancymath.cpp
  core/fileutils.cpp
  core/hash.cpp
  core/logger.cpp
  core/mainargs.cpp
  core/makedir.cpp
  core/md5.cpp
  core/multithread.cpp
  core/parallel.cpp
  core/rand.cpp
  core/rand_helpers.cpp
  core/sha2.cpp
  core/test.cpp
  core/threadsafecounter.cpp
  core/threadsafequeue.cpp
  core/threadtest.cpp
  core/timer.cpp
  game/board.cpp
  game/rules.cpp
  game/boardhistory.cpp
  game/graphhash.cpp
  dataio/sgf.cpp
  dataio/numpywrite.cpp
  dataio/poswriter.cpp
  dataio/trainingwrite.cpp
  dataio/loadmodel.cpp
  dataio/homedata.cpp
  dataio/files.cpp
  neuralnet/nninputs.cpp
  neuralnet/sgfmetadata.cpp
  neuralnet/modelversion.cpp
  neuralnet/nneval.cpp
  neuralnet/desc.cpp
  neuralnet/debugprint.cpp
  ${NEURALNET_BACKEND_SOURCES}
  book/book.cpp
  book/bookcssjs.cpp
  search/timecontrols.cpp
  search/searchparams.cpp
  search/mutexpool.cpp
  search/search.cpp
  search/searchnode.cpp
  search/searchresults.cpp
  search/searchhelpers.cpp
  search/searchexplorehelpers.cpp
  search/searchmirror.cpp
  search/searchmultithreadhelpers.cpp
  search/searchnnhelpers.cpp
  search/searchtimehelpers.cpp
  search/searchupdatehelpers.cpp
  search/asyncbot.cpp
  search/distributiontable.cpp
  search/localpattern.cpp
  search/searchnodetable.cpp
  search/subtreevaluebiastable.cpp
  search/evalcache.cpp
  search/patternbonustable.cpp
  search/analysisdata.cpp
  search/reportedsearchvalues.cpp
  program/gtpconfig.cpp
  program/setup.cpp
  program/playutils.cpp
  program/playsettings.cpp
  program/play.cpp
  program/selfplaymanager.cpp
  ${GIT_HEADER_FILE_ALWAYS_UPDATED}
  tests/testboardarea.cpp
  tests/testboardbasic.cpp
  tests/testbook.cpp
  tests/testcommon.cpp
  tests/testconfig.cpp
  tests/testmisc.cpp
  tests/testnnevalcanary.cpp
  tests/testrules.cpp
  tests/testscore.cpp
  tests/testsgf.cpp
  tests/testsymmetries.cpp
  tests/testnninputs.cpp
  tests/testownership.cpp
  tests/testsearchcommon.cpp
  tests/testsearchnonn.cpp
  tests/testsearch.cpp
  tests/testsearchv3.cpp
  tests/testsearchv8.cpp
  tests/testsearchv9.cpp
  tests/testsearchmisc.cpp
  tests/testtime.cpp
  tests/testtrainingwrite.cpp
  tests/testnn.cpp
  tests/tinymodel.cpp
  tests/tinymodeldata.cpp
  distributed/client.cpp
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
  command/contribute.cpp
  command/evalsgf.cpp
  command/gatekeeper.cpp
  command/genbook.cpp
  command/gputest.cpp
  command/gtp.cpp
  command/match.cpp
  command/misc.cpp
  command/runtests.cpp
  command/sandbox.cpp
  command/selfplay.cpp
  command/startposes.cpp
  command/tune.cpp
  command/writetrainingdata.cpp
  main.cpp
  )

if(USE_BACKEND STREQUAL "CUDA")
  target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
  target_compile_definitions(katago PRIVATE CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  find_package(CUDAToolkit REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  include_directories(SYSTEM external/cudnn-frontend/include) #SYSTEM suppresses warnings inside the vendored header-only cudnn_frontend library
  # cudnn_frontend's experimental OSS engine headers reference NVRTC symbols at link time
  # (we don't actually use those engines, but the references are pulled in by the header-only includes).
  target_link_libraries(katago CUDA::cublas ${CUDNN_LIBRARY} CUDA::nvrtc)
elseif(USE_BACKEND STREQUAL "TENSORRT")
  target_compile_definitions(katago PRIVATE USE_TENSORRT_BACKEND)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
  if((NOT TENSORRT_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} NvInfer.h was NOT found, specify TENSORRT_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()

  # Hackily extract out the version from the TensorRT header
  # In each case, try the old format and on failure try the new format.
  file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version_header)

  set(TENSORRT_VERSION_MAJOR "")
  set(TENSORRT_VERSION_MINOR "")
  set(TENSORRT_VERSION_PATCH "")

  string(REGEX MATCH "#define NV_TENSORRT_MAJOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MAJOR_ENTERPRISE +([0-9]+)" tensorrt_major_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_MINOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MINOR_ENTERPRISE +([0-9]+)" tensorrt_minor_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_PATCH +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_PATCH_ENTERPRISE +([0-9]+)" tensorrt_patch_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
  endif()

  set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
  message(STATUS "Detected TensorRT version: ${TENSORRT_VERSION}")

  if("${TENSORRT_VERSION_MAJOR}" STREQUAL "" OR "${TENSORRT_VERSION_MINOR}" STREQUAL "" OR "${TENSORRT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Could not determine TensorRT version from header file")
  endif()

  find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_${TENSORRT_VERSION_MAJOR} HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)

  # Version 8 is required for serializing the builder timing cache.
  # Version 8.2 is required for eliminating the global logger for Builder and Runtime.
  # Version 8.5 is required for eliminating many deprecated APIs and adopting new features.
  # Version 8.6 is for CUDA 12 support and further reduction in initialization time.
  if(TENSORRT_VERSION VERSION_LESS 8.5)
    message(FATAL_ERROR "TensorRT 8.5 or greater is required but ${TENSORRT_VERSION} was found.")
  endif()
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  target_link_libraries(katago CUDA::cudart_static ${TENSORRT_LIBRARY})

  # The TensorRT backend builds networks by emitting an ONNX ModelProto and handing the
  # serialized bytes to TensorRT's nvonnxparser. We compile the vendored ONNX schema
  # (cpp/external/onnx/onnx.proto) with protoc and link against our own protobuf; this is
  # symbol-isolated from the protobuf that TensorRT statically links into nvonnxparser, so
  # there is no ABI contact between the two (the handoff is serialized bytes).
  find_package(Protobuf REQUIRED)
  message(STATUS "Found Protobuf version: ${Protobuf_VERSION}")
  set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx")
  protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS "${ONNX_PROTO_DIR}/onnx.proto")
  # protoc-generated code is not ours to lint; silence its warnings to keep build output readable.
  set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES COMPILE_OPTIONS "-w")
  target_sources(katago PRIVATE ${ONNX_PROTO_SRCS} neuralnet/onnxmodelbuilder.cpp)
  # Generated onnx.pb.h lands in the build dir; let backend code include it.
  target_include_directories(katago SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})

  # nvonnxparser ships alongside nvinfer in the TensorRT lib dir. Derive that dir from the
  # already-resolved TENSORRT_LIBRARY so this works both with TENSORRT_ROOT_DIR and with a
  # directly-specified -DTENSORRT_LIBRARY
  get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
  find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_${TENSORRT_VERSION_MAJOR} HINTS ${TENSORRT_ROOT_DIR} ${TENSORRT_LIBRARY_DIR} PATH_SUFFIXES lib)
  if(NOT TENSORRT_ONNXPARSER_LIBRARY)
    message(FATAL_ERROR "${ColorBoldRed} libnvonnxparser was NOT found in the TensorRT lib dir. ${ColorReset}")
  endif()
  target_link_libraries(katago ${TENSORRT_ONNXPARSER_LIBRARY} ${Protobuf_LIBRARIES})
elseif(USE_BACKEND STREQUAL "METAL")
  target_compile_definitions(katago PRIVATE USE_METAL_BACKEND)
  target_link_libraries(katago KataGoSwift katagocoreml
    ${KATAGOCOREML_DEP_LDFLAGS}
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph")
  if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    message(WARNING "Metal backend may not work optimally on Intel. ARM64 architecture is recommended.")
  endif()
elseif(USE_BACKEND STREQUAL "OPENCL")
  target_compile_definitions(katago PRIVATE USE_OPENCL_BACKEND)
  find_package(OpenCL)
  if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL not found, attempting to see if CUDA exists and has OpenCL since sometimes CUDA may provide OpenCL where cmake can't find it.")
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "OpenCL installation not found")
    else()
      message(WARNING "OpenCL not found, but found CUDA, attempting to use OpenCL via CUDA.")
    endif()
    include_directories(${OpenCL_INCLUDE_DIRS})
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(katago CUDA::OpenCL)
  else()
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
    target_link_libraries(katago ${OpenCL_LIBRARY})
  endif()
elseif(USE_BACKEND STREQUAL "EIGEN")
  target_compile_definitions(katago PRIVATE USE_EIGEN_BACKEND)
  # Allow EIGEN3_INCLUDE_DIRS as a manual override (for users who downloaded
  # Eigen3 and just want to point at the unpacked directory).
  if(EIGEN3_INCLUDE_DIRS)
    message(STATUS "Using user-provided EIGEN3_INCLUDE_DIRS=${EIGEN3_INCLUDE_DIRS}")
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
  else()
    if(NOT (MSVC))
      find_package(Eigen3 REQUIRED)
    else()
      find_package(Eigen3)
      if(NOT Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} CACHE PATH "Directory containing 'Eigen' and 'unsupported' subdirs with Eigen headers")
        message(FATAL_ERROR "*** KataGo: Eigen3 was not found. If you've downloaded Eigen3, you can IGNORE the whole above error from CMake and just set 'EIGEN3_INCLUDE_DIRS' to the root eigen3-3.*.* directory after unzipping it. Among other things it should contain 'Eigen' and 'unsupported' subdirs, which is all KataGo needs. Otherwise, you will need to 'build' Eigen3 even though it is header-only and install it via MSVC in order for cmake to find it via FindEigen3.cmake.")
      endif()
    endif()
    # Modern Eigen3Config.cmake (Homebrew, apt, vcpkg, ...) only exports the
    # imported target Eigen3::Eigen with INTERFACE_INCLUDE_DIRECTORIES set; the
    # legacy variable EIGEN3_INCLUDE_DIRS is not always populated. Prefer the
    # target where available, fall back to the variable for older installs.
    if(TARGET Eigen3::Eigen)
      # Plain (non-keyword) form to match the rest of KataGo's CMakeLists,
      # which doesn't use PRIVATE/PUBLIC/INTERFACE on target_link_libraries.
      target_link_libraries(katago Eigen3::Eigen)
      message(STATUS "Found Eigen3 (using Eigen3::Eigen imported target)")
    else()
      include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
      message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
    endif()
  endif()
elseif(USE_BACKEND STREQUAL "VULKAN")
  target_compile_definitions(katago PRIVATE USE_VULKAN_BACKEND)
  if(KATAGO_HAS_COOPMAT_SHADERS)
    target_compile_definitions(katago PRIVATE KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    if(KATAGO_HAS_COOPMAT_MAINTENANCE1_SHADERS)
      target_compile_definitions(katago PRIVATE KATAGO_VULKAN_HAS_COOPMAT_MAINTENANCE1_SHADERS)
    endif()
  endif()
  if(KATAGO_HAS_COOPMAT2_SHADERS)
    target_compile_definitions(katago PRIVATE KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
  endif()
  include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(katago ${Vulkan_LIBRARIES})
endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=50)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  target_link_libraries(katago ${ZLIB_LIBRARIES})
else()
  set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR} CACHE PATH "Path to directory with zlib.h and other header files")
  set(ZLIB_LIBRARY ${ZLIB_LIBRARY} CACHE FILEPATH "Path to 'libz.so' on Linux or 'libz.lib' on Windows")
  mark_as_advanced(CLEAR ZLIB_INCLUDE_DIR ZLIB_LIBRARY)
  message(SEND_ERROR "${ColorBoldRed}zlib was not found, if zlib is actually installed but not being found you can set ZLIB_INCLUDE_DIR to the directory with zlib.h and other headers, and ZLIB_LIBRARY to the compiled library 'libz.so' on Linux or 'libz.lib' on Windows. On the command line, this is -DZLIB_INCLUDE_DIR=... and -DZLIB_LIBRARY=... ${ColorReset}")
endif(ZLIB_FOUND)

find_library(LIBZIP_LIBRARY NAMES zip)
find_path(LIBZIP_INCLUDE_DIR_ZIP NAMES zip.h)
find_path(LIBZIP_INCLUDE_DIR_ZIPCONF NAMES zipconf.h)
if((NOT LIBZIP_LIBRARY) OR (NOT LIBZIP_INCLUDE_DIR_ZIP) OR (NOT LIBZIP_INCLUDE_DIR_ZIPCONF))
  if(BUILD_DISTRIBUTED)
    message(SEND_ERROR "${ColorBoldRed}WARNING: BUILD_DISTRIBUTED was requested but libzip library was NOT found. KataGo needs this for writing training data so libzip is required. On Linux, install through your normal package manager. On Windows, set LIBZIP_INCLUDE_DIR_ZIP to the directory that includes zip.h and other files, and LIBZIP_INCLUDE_DIR_ZIPCONF to the directory that includes zipconf.h and other files, and LIBZIP_LIBRARY to the libzip.lib or zip.lib file. ${ColorReset}")
  endif()
  target_compile_definitions(katago PRIVATE NO_LIBZIP)
  message(WARNING "${ColorBoldRed}WARNING: libzip library was NOT found. KataGo should still work for GTP/matches/analysis if everything else is good, but selfplay for writing training data will not be possible.${ColorReset}")
  set(LIBZIP_INCLUDE_DIR_ZIP ${LIBZIP_INCLUDE_DIR_ZIP} CACHE PATH "Path to directory with zip.h and other header files")
  set(LIBZIP_INCLUDE_DIR_ZIPCONF ${LIBZIP_INCLUDE_DIR_ZIPCONF} CACHE PATH "Path to directory with zipconf.h and other header files")
  set(LIBZIP_LIBRARY ${LIBZIP_LIBRARY} CACHE FILEPATH "Path to 'libzip.so' on Linux or 'libzip.lib' or 'zip.lib' on Windows")
  mark_as_advanced(CLEAR LIBZIP_INCLUDE_DIR_ZIP LIBZIP_INCLUDE_DIR_ZIPCONF LIBZIP_LIBRARY)
else()
  include_directories(${LIBZIP_INCLUDE_DIR_ZIP})
  include_directories(${LIBZIP_INCLUDE_DIR_ZIPCONF})
  target_link_libraries(katago ${LIBZIP_LIBRARY})
endif()

if(USE_TCMALLOC)
  target_link_libraries(katago ${TCMALLOC_LIB})
endif(USE_TCMALLOC)

if(BUILD_DISTRIBUTED)
  message(STATUS "-DBUILD_DISTRIBUTED=1 is set, compiling code and dependencies to contribute to distributed training")
  target_compile_definitions(katago PRIVATE BUILD_DISTRIBUTED)
  find_package(OpenSSL REQUIRED)
  target_link_libraries(katago ${OPENSSL_SSL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARIES})
  include_directories(${OPENSSL_INCLUDE_DIR})
  include_directories(external/httplib)
endif()

#------------------------------------------------------------------------------------

if (WIN32) # handles both x86 and x64
  message(STATUS "Setting up build for Windows.")
  # Suppress min and max macros on windows
  # Also define a few other things
  target_compile_definitions(katago PRIVATE NOMINMAX)
  target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
  target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
  target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
  # core/rand.cpp uses winsock for a gethostname
  target_link_libraries(katago ws2_32)
  if (MINGW)
    message(STATUS "Setting up build for mingw.")
    if(USE_BACKEND STREQUAL "CUDA" OR USE_BACKEND STREQUAL "TENSORRT")
      message(SEND_ERROR "CUDA and TENSORRT MinGW backends are not supported by NVIDIA, see https://forums.developer.nvidia.com/t/cuda-with-mingw-how-to-get-cuda-running-under-mingw")
    endif()

    # Make the build independent of MinGW dynamic libraries
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      # Strip symbols. It significantly reduces binary size (makes it 20x smaller)
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
    endif()

    # It's needed for linking `PathRemoveFileSpecW` and probably other Windows API functions
    target_link_libraries(katago shlwapi)

    # For some reason MINGW doesn't define `HAVE_LONG_LONG` directive on his own unlike MSVC
    # Define it because TCLAP library needs it. Actually it's part of standard and should be supported everywhere
    include(CheckTypeSize)
    check_type_size("long long" HAVE_LONG_LONG)
    if(HAVE_LONG_LONG)
      target_compile_definitions(katago PRIVATE HAVE_LONG_LONG)
    endif()
  endif()
endif()

if(MSVC)
  message(STATUS "Setting up build for MSVC.")
  # Go ahead and suppress some MSVC warnings about sprintf and similar
  # things. They might be useful in some cases, but also are noisy.
  target_compile_definitions(katago PRIVATE _CRT_SECURE_NO_WARNINGS)

  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 -D__FMA__")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Setting up build for GNU, Clang or MinGW.")
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)"))
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
  else()
    # For ARM architecture, as a hack, ensure that char is signed
    message(STATUS "ARM architecture detected: adding -fsigned-char flag")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
  endif()
  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  find_package (Threads REQUIRED)
  target_link_libraries(katago Threads::Threads)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wunused")
  set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

  if(NOT CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
    message(STATUS "CXX flags from build type: ${CMAKE_CXX_FLAGS_${_bt}}")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Enabling GNU-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op -Wnoexcept -Wstrict-null-sentinel")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wrestrict")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Enabling AppleClang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else -Wno-deprecated-declarations -Wno-nullability-extension -Wno-cast-qual -Wno-c11-extensions")
  else()
    message(STATUS "Enabling Clang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else")
    target_link_libraries(katago "atomic")
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  endif()

endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#ifndef NEURALNET_VULKAN_INCLUDES_H_
#define NEURALNET_VULKAN_INCLUDES_H_

// Do NOT define VK_NO_PROTOTYPES — we want the function prototypes provided by vulkan.h.
#include <vulkan/vulkan.h>

// Require Vulkan 1.1+ headers (needed for subgroups, push descriptors, etc.)
#if !defined(VK_API_VERSION_1_1)
#error "Vulkan headers 1.1+ required"
#endif

// Subgroup-size-control started as VK_EXT_subgroup_size_control (usable on
// pre-1.3 Vulkan API versions) and was promoted to core in Vulkan 1.3.
// These aliases let one codepath compile against both header vintages:
// older headers exposing only EXT spellings, and newer headers exposing core
// 1.3 names.
#ifndef VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
#define VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME "VK_EXT_subgroup_size_control"
#endif

#if !defined(VK_VERSION_1_3) && defined(VK_EXT_subgroup_size_control)
typedef VkPhysicalDeviceSubgroupSizeControlFeaturesEXT VkPhysicalDeviceSubgroupSizeControlFeatures;
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES \
  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT
typedef VkPhysicalDeviceSubgroupSizeControlPropertiesEXT VkPhysicalDeviceSubgroupSizeControlProperties;
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES \
  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT
#define VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT \
  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
typedef VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT VkPipelineShaderStageRequiredSubgroupSizeCreateInfo;
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO \
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT
#endif

#if defined(VK_VERSION_1_3) || defined(VK_EXT_subgroup_size_control)
#define KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS 1
#endif

#ifndef VK_KHR_16BIT_STORAGE_EXTENSION_NAME
#define VK_KHR_16BIT_STORAGE_EXTENSION_NAME "VK_KHR_16bit_storage"
#endif

#ifndef VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
#define VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME "VK_KHR_shader_float16_int8"
#endif

#if defined(VK_VERSION_1_2) || defined(VK_KHR_shader_float16_int8)
#define KATAGO_HAS_SHADER_FLOAT16_INT8_HEADERS 1
#endif

#ifndef VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
#define VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME "VK_KHR_shader_float_controls"
#endif

#if defined(VK_VERSION_1_2) || defined(VK_KHR_shader_float_controls)
#define KATAGO_HAS_SHADER_FLOAT_CONTROLS_HEADERS 1
#endif

#ifndef VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME
#define VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME "VK_KHR_vulkan_memory_model"
#endif

#if defined(VK_VERSION_1_2) || defined(VK_KHR_vulkan_memory_model)
#define KATAGO_HAS_VULKAN_MEMORY_MODEL_HEADERS 1
#endif

// VK_KHR_cooperative_matrix landed in header vintages after some we support.
// Key off the extension's own version macro (a real #define), NOT off enum
// constants like VK_STRUCTURE_TYPE_* (defined() is meaningless on enum values).
// When absent, vulkanhelpers.cpp provides locally-mirrored spec types.
#if defined(VK_KHR_cooperative_matrix)
#define KATAGO_HAS_COOPERATIVE_MATRIX_HEADERS 1
#endif

#if defined(VK_NV_cooperative_matrix2)
#define KATAGO_HAS_COOPERATIVE_MATRIX_2_HEADERS 1
#endif
#ifndef VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
#define VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME "VK_NV_cooperative_matrix2"
#endif

#endif  // NEURALNET_VULKAN_INCLUDES_H_

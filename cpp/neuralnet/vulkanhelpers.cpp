#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkanhelpers.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "../core/global.h"
#include "../core/test.h"
#include "vulkanshaders_generated.h"
#include "vulkantuner.h"

using namespace std;
using half_t = half_float::half;

// VulkanHelpers::checkResult

namespace {

  string_view vkResultString(VkResult r) {
    switch(r) {
      case VK_SUCCESS:
        return "VK_SUCCESS";
      case VK_NOT_READY:
        return "VK_NOT_READY";
      case VK_TIMEOUT:
        return "VK_TIMEOUT";
      case VK_EVENT_SET:
        return "VK_EVENT_SET";
      case VK_EVENT_RESET:
        return "VK_EVENT_RESET";
      case VK_INCOMPLETE:
        return "VK_INCOMPLETE";
      case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
      case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
      case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
      case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
      case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
      case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
      case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
      case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
      case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
      case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
      case VK_ERROR_FORMAT_NOT_SUPPORTED:
        return "VK_ERROR_FORMAT_NOT_SUPPORTED";
      case VK_ERROR_FRAGMENTED_POOL:
        return "VK_ERROR_FRAGMENTED_POOL";
      default:
        return "VK_UNKNOWN_ERROR";
    }
  }

  // VK_VALVE_shader_mixed_float_dot_product feature struct.
  // The stock Vulkan headers do not yet expose this type, so we mirror the spec
  // layout locally. Used for both feature detection and vkCreateDevice enablement.
  struct VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local {
    VkStructureType sType;
    void* pNext;
    VkBool32 shaderMixedFloatDotProductFloat16AccFloat32;
    VkBool32 shaderMixedFloatDotProductFloat16AccFloat16;
    VkBool32 shaderMixedFloatDotProductBFloat16Acc;
    VkBool32 shaderMixedFloatDotProductFloat8AccFloat32;
  };
  // Guard against layout drift if the official struct lands in Vulkan headers
  // later. The struct is: 4B sType + (pointer-aligned) void* pNext + 4x VkBool32.
  // On 64-bit builds the compiler inserts 4B of padding after sType; on 32-bit
  // builds there is no padding. Compute the expectation from offsetof rather
  // than sum-of-sizeof so the assert is portable.
  static_assert(
    offsetof(
      VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local,
      shaderMixedFloatDotProductFloat16AccFloat32) ==
      offsetof(VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local, pNext) + sizeof(void*),
    "Valve dot2 feature struct layout drift");
  static_assert(
    sizeof(VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local) ==
      offsetof(
        VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local,
        shaderMixedFloatDotProductFloat16AccFloat32) +
        4 * sizeof(VkBool32),
    "Valve dot2 feature struct trailing size drift");
  constexpr VkStructureType kVkStructureTypeShaderMixedFloatDotProductFeaturesVALVE =
    static_cast<VkStructureType>(1000673000);

  // VK_KHR_vulkan_memory_model header-compatibility shim. The coopmat shaders
  // declare OpCapability VulkanMemoryModel, so this feature struct is needed
  // even when compiling against Vulkan 1.1-era headers.
#if defined(KATAGO_HAS_VULKAN_MEMORY_MODEL_HEADERS)
  using VulkanMemoryModelFeatures = VkPhysicalDeviceVulkanMemoryModelFeatures;
  constexpr VkStructureType kSTypeVulkanMemoryModelFeatures =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
  constexpr const char* kVulkanMemoryModelExtensionName = VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME;
#else
  struct VulkanMemoryModelFeatures {
    VkStructureType sType;
    void* pNext;
    VkBool32 vulkanMemoryModel;
    VkBool32 vulkanMemoryModelDeviceScope;
    VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
  };
  constexpr VkStructureType kSTypeVulkanMemoryModelFeatures = static_cast<VkStructureType>(1000211000);
  constexpr const char* kVulkanMemoryModelExtensionName = "VK_KHR_vulkan_memory_model";
#endif

  // VK_KHR_shader_float_controls / Vulkan 1.2 property struct. Some supported
  // header vintages may lack the type, so mirror the spec layout when needed.
#if defined(KATAGO_HAS_SHADER_FLOAT_CONTROLS_HEADERS)
  using FloatControlsProperties = VkPhysicalDeviceFloatControlsProperties;
  constexpr VkStructureType kSTypeFloatControlsProperties = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
#else
  struct FloatControlsProperties {
    VkStructureType sType;
    void* pNext;
    int32_t denormBehaviorIndependence;
    int32_t roundingModeIndependence;
    VkBool32 shaderSignedZeroInfNanPreserveFloat16;
    VkBool32 shaderSignedZeroInfNanPreserveFloat32;
    VkBool32 shaderSignedZeroInfNanPreserveFloat64;
    VkBool32 shaderDenormPreserveFloat16;
    VkBool32 shaderDenormPreserveFloat32;
    VkBool32 shaderDenormPreserveFloat64;
    VkBool32 shaderDenormFlushToZeroFloat16;
    VkBool32 shaderDenormFlushToZeroFloat32;
    VkBool32 shaderDenormFlushToZeroFloat64;
    VkBool32 shaderRoundingModeRTEFloat16;
    VkBool32 shaderRoundingModeRTEFloat32;
    VkBool32 shaderRoundingModeRTEFloat64;
    VkBool32 shaderRoundingModeRTZFloat16;
    VkBool32 shaderRoundingModeRTZFloat32;
    VkBool32 shaderRoundingModeRTZFloat64;
  };
  constexpr VkStructureType kSTypeFloatControlsProperties = static_cast<VkStructureType>(1000197000);
#endif
  constexpr const char* kShaderFloatControlsExtensionName = VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;

  // VK_KHR_cooperative_matrix header-compatibility shim.
  //
  // KATAGO_HAS_COOPERATIVE_MATRIX_HEADERS is defined in vulkanincludes.h off the
  // extension's version macro (a real #define). Do NOT test availability with
  // defined(VK_STRUCTURE_TYPE_...) — those are enum *constants*, not macros, so
  // defined() is always false on them. When the headers lack the types we fall
  // back to a locally-mirrored spec layout.
#if defined(KATAGO_HAS_COOPERATIVE_MATRIX_HEADERS)
  using CoopmatFeaturesKHR = VkPhysicalDeviceCooperativeMatrixFeaturesKHR;
  using CoopmatPropertiesKHR = VkCooperativeMatrixPropertiesKHR;
  using CoopmatDevicePropertiesKHR = VkPhysicalDeviceCooperativeMatrixPropertiesKHR;
  using CoopmatComponentType = VkComponentTypeKHR;
  using CoopmatScope = VkScopeKHR;
  using PFN_GetCoopmatProps = PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
  constexpr VkStructureType kSTypeCoopmatFeatures = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  constexpr VkStructureType kSTypeCoopmatProperties = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  constexpr VkStructureType kSTypeCoopmatDeviceProperties =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  constexpr CoopmatComponentType kComponentFloat16 = VK_COMPONENT_TYPE_FLOAT16_KHR;
  constexpr CoopmatComponentType kComponentFloat32 = VK_COMPONENT_TYPE_FLOAT32_KHR;
  constexpr CoopmatScope kScopeSubgroup = VK_SCOPE_SUBGROUP_KHR;
  constexpr const char* kCoopmatExtensionName = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
#else
  // Mirror of the VK_KHR_cooperative_matrix spec types for headers that lack them.
  struct CoopmatFeaturesKHR {
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
  };
  using CoopmatComponentType = int32_t;
  using CoopmatScope = int32_t;
  struct CoopmatPropertiesKHR {
    VkStructureType sType;
    void* pNext;
    uint32_t MSize;
    uint32_t NSize;
    uint32_t KSize;
    CoopmatComponentType AType;
    CoopmatComponentType BType;
    CoopmatComponentType CType;
    CoopmatComponentType ResultType;
    VkBool32 saturatingAccumulation;
    CoopmatScope scope;
  };
  // VkPhysicalDeviceCooperativeMatrixPropertiesKHR — carries the stage mask that
  // tells us whether coopmat is usable from a compute shader (queried via
  // vkGetPhysicalDeviceProperties2, distinct from the per-shape enumeration above).
  struct CoopmatDevicePropertiesKHR {
    VkStructureType sType;
    void* pNext;
    VkShaderStageFlags cooperativeMatrixSupportedStages;
  };
  typedef VkResult(VKAPI_PTR* PFN_GetCoopmatProps)(VkPhysicalDevice, uint32_t*, CoopmatPropertiesKHR*);
  constexpr VkStructureType kSTypeCoopmatFeatures = static_cast<VkStructureType>(1000506000);
  constexpr VkStructureType kSTypeCoopmatProperties = static_cast<VkStructureType>(1000506001);
  constexpr VkStructureType kSTypeCoopmatDeviceProperties = static_cast<VkStructureType>(1000506002);
  constexpr CoopmatComponentType kComponentFloat16 = 0;
  constexpr CoopmatComponentType kComponentFloat32 = 1;
  constexpr CoopmatScope kScopeSubgroup = 3;
  constexpr const char* kCoopmatExtensionName = "VK_KHR_cooperative_matrix";
#endif
  constexpr CoopmatScope kScopeWorkgroup = static_cast<CoopmatScope>(2);

  // VK_EXT_cooperative_matrix_maintenance1 is newer than several Vulkan
  // header versions KataGo supports. Mirror its feature struct when necessary;
  // its ABI is the spec-defined VkStructureType, pNext, then five VkBool32s.
#if defined(VK_EXT_cooperative_matrix_maintenance1)
  using CoopmatMaintenance1FeaturesEXT = VkPhysicalDeviceCooperativeMatrixMaintenance1FeaturesEXT;
  constexpr VkStructureType kSTypeCoopmatMaintenance1Features =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_MAINTENANCE_1_FEATURES_EXT;
  constexpr const char* kCoopmatMaintenance1ExtensionName = VK_EXT_COOPERATIVE_MATRIX_MAINTENANCE_1_EXTENSION_NAME;
#else
  struct CoopmatMaintenance1FeaturesEXT {
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrixProperties2;
    VkBool32 cooperativeMatrixReductions;
    VkBool32 cooperativeMatrixConversions;
    VkBool32 cooperativeMatrixPerElementOperations;
    VkBool32 cooperativeMatrixGetCoordinate;
  };
  constexpr VkStructureType kSTypeCoopmatMaintenance1Features = static_cast<VkStructureType>(1000659000);
  constexpr const char* kCoopmatMaintenance1ExtensionName = "VK_EXT_cooperative_matrix_maintenance1";
#endif

  // VK_NV_cooperative_matrix2 header-compatibility shim.
#if defined(KATAGO_HAS_COOPERATIVE_MATRIX_2_HEADERS)
  using Coopmat2FeaturesNV = VkPhysicalDeviceCooperativeMatrix2FeaturesNV;
  using Coopmat2DevicePropertiesNV = VkPhysicalDeviceCooperativeMatrix2PropertiesNV;
  using Coopmat2FlexPropertiesNV = VkCooperativeMatrixFlexibleDimensionsPropertiesNV;
  using PFN_GetCoopmat2FlexProps = PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV;
  constexpr VkStructureType kSTypeCoopmat2Features = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
  constexpr VkStructureType kSTypeCoopmat2DeviceProperties =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;
  constexpr VkStructureType kSTypeCoopmat2FlexProperties =
    VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
  constexpr const char* kCoopmat2ExtensionName = VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME;
#else
  struct Coopmat2FeaturesNV {
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrixWorkgroupScope;
    VkBool32 cooperativeMatrixFlexibleDimensions;
    VkBool32 cooperativeMatrixReductions;
    VkBool32 cooperativeMatrixConversions;
    VkBool32 cooperativeMatrixPerElementOperations;
    VkBool32 cooperativeMatrixTensorAddressing;
    VkBool32 cooperativeMatrixBlockLoads;
  };
  struct Coopmat2DevicePropertiesNV {
    VkStructureType sType;
    void* pNext;
    uint32_t cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
    uint32_t cooperativeMatrixFlexibleDimensionsMaxDimension;
    uint32_t cooperativeMatrixWorkgroupScopeReservedSharedMemory;
  };
  struct Coopmat2FlexPropertiesNV {
    VkStructureType sType;
    void* pNext;
    uint32_t MGranularity;
    uint32_t NGranularity;
    uint32_t KGranularity;
    CoopmatComponentType AType;
    CoopmatComponentType BType;
    CoopmatComponentType CType;
    CoopmatComponentType ResultType;
    VkBool32 saturatingAccumulation;
    CoopmatScope scope;
    uint32_t workgroupInvocations;
  };
  typedef VkResult(VKAPI_PTR* PFN_GetCoopmat2FlexProps)(VkPhysicalDevice, uint32_t*, Coopmat2FlexPropertiesNV*);
  constexpr VkStructureType kSTypeCoopmat2Features = static_cast<VkStructureType>(1000593000);
  constexpr VkStructureType kSTypeCoopmat2FlexProperties = static_cast<VkStructureType>(1000593001);
  constexpr VkStructureType kSTypeCoopmat2DeviceProperties = static_cast<VkStructureType>(1000593002);
  constexpr const char* kCoopmat2ExtensionName = "VK_NV_cooperative_matrix2";
#endif

  [[maybe_unused]] bool coopmat2RequiredFeaturesSupported(const Coopmat2FeaturesNV& feats) {
    // Every coopmat2 GEMM/conv/winograd/attention shader loads and stores
    // cooperative matrices through tensor-addressing layouts, so the tensor
    // addressing feature is a hard requirement for the whole coopmat2 path.
    // The attention-only features (reductions, conversions, per-element
    // operations, block loads) are gated separately via supportsCoopmat2Attention.
    return feats.cooperativeMatrixWorkgroupScope == VK_TRUE && feats.cooperativeMatrixFlexibleDimensions == VK_TRUE &&
           feats.cooperativeMatrixTensorAddressing == VK_TRUE;
  }

  struct VulkanGemmVariantFilter {
    bool allowCoopmat2 = true;
    bool allowCoopmat1 = true;
    bool allowDot2 = true;
    bool allowAccF32 = true;
    bool allowAccF16 = true;
    bool allowCoopmat2AccF32 = true;
    bool allowCoopmat2AccF16 = true;
    bool allowCoopmat1AccF32 = true;
    bool allowCoopmat1AccF16 = true;
    bool allowDot2AccF32 = true;
    // FP16 Dot2 accumulation is experimental and must be selected explicitly
    // via accf16 or dot2accf16 in KATAGO_VULKAN_ACCEL_VARIANTS.
    bool allowDot2AccF16 = false;
    bool configured = false;
  };

  enum class GemmVariantTokenKind {
    Unknown,
    Coopmat,
    Coopmat2,
    Coopmat1,
    Dot2,
    AccF16,
    AccF32,
    Coopmat2AccF32,
    Coopmat2AccF16,
    Coopmat1AccF32,
    Coopmat1AccF16,
    Dot2AccF32,
    Dot2AccF16
  };

  struct ParsedGemmVariantToken {
    GemmVariantTokenKind kind;
    bool disable;
    string_view original;
  };

  vector<string_view> splitEnvTokens(string_view value) {
    vector<string_view> tokens;
    size_t tokenStart = string_view::npos;
    for(size_t i = 0; i < value.size(); i++) {
      char c = value[i];
      if(c == ',' || c == ';' || c == ':' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        if(tokenStart != string_view::npos) {
          tokens.push_back(value.substr(tokenStart, i - tokenStart));
          tokenStart = string_view::npos;
        }
      } else if(tokenStart == string_view::npos)
        tokenStart = i;
    }
    if(tokenStart != string_view::npos)
      tokens.push_back(value.substr(tokenStart));
    return tokens;
  }

  bool isSpecificGemmVariantToken(GemmVariantTokenKind kind) {
    return kind == GemmVariantTokenKind::Coopmat2AccF32 || kind == GemmVariantTokenKind::Coopmat2AccF16 ||
           kind == GemmVariantTokenKind::Coopmat1AccF32 || kind == GemmVariantTokenKind::Coopmat1AccF16 ||
           kind == GemmVariantTokenKind::Dot2AccF32 || kind == GemmVariantTokenKind::Dot2AccF16;
  }

  bool isGemmFamilyToken(GemmVariantTokenKind kind) {
    return kind == GemmVariantTokenKind::Coopmat || kind == GemmVariantTokenKind::Coopmat2 ||
           kind == GemmVariantTokenKind::Coopmat1 || kind == GemmVariantTokenKind::Dot2;
  }

  bool isGemmAccumulatorToken(GemmVariantTokenKind kind) {
    return kind == GemmVariantTokenKind::AccF16 || kind == GemmVariantTokenKind::AccF32;
  }

  ParsedGemmVariantToken parseGemmVariantToken(string_view token) {
    string_view raw = token;
    bool disable = false;
    if(!raw.empty() && raw.front() == '-') {
      disable = true;
      raw.remove_prefix(1);
    }

    GemmVariantTokenKind kind = GemmVariantTokenKind::Unknown;
    if(raw == "coopmat")
      kind = GemmVariantTokenKind::Coopmat;
    else if(raw == "coopmat2")
      kind = GemmVariantTokenKind::Coopmat2;
    else if(raw == "coopmat1")
      kind = GemmVariantTokenKind::Coopmat1;
    else if(raw == "dot2")
      kind = GemmVariantTokenKind::Dot2;
    else if(raw == "accf16")
      kind = GemmVariantTokenKind::AccF16;
    else if(raw == "accf32")
      kind = GemmVariantTokenKind::AccF32;
    else if(raw == "coopmat2accf32")
      kind = GemmVariantTokenKind::Coopmat2AccF32;
    else if(raw == "coopmat2accf16")
      kind = GemmVariantTokenKind::Coopmat2AccF16;
    else if(raw == "coopmat1accf32")
      kind = GemmVariantTokenKind::Coopmat1AccF32;
    else if(raw == "coopmat1accf16")
      kind = GemmVariantTokenKind::Coopmat1AccF16;
    else if(raw == "dot2accf32")
      kind = GemmVariantTokenKind::Dot2AccF32;
    else if(raw == "dot2accf16")
      kind = GemmVariantTokenKind::Dot2AccF16;

    return ParsedGemmVariantToken{kind, disable, token};
  }

  // One row per concrete accelerator GEMM variant (a specific family x
  // accumulator combination). The family/accumulator/specific member pointers
  // are the three filter gates whose AND decides whether the variant is usable;
  // token is its KATAGO_VULKAN_ACCEL_VARIANTS name and disabledBits() the
  // tuned-kernel mask to clear when the variant is filtered out.
  struct GemmVariantDescriptor {
    GemmVariantTokenKind kind;
    bool VulkanGemmVariantFilter::* familyField;
    bool VulkanGemmVariantFilter::* accField;
    bool VulkanGemmVariantFilter::* specificField;
    const char* token;
    int64_t (*disabledBits)();
  };

  const GemmVariantDescriptor kGemmVariantDescriptors[] = {
    {GemmVariantTokenKind::Coopmat2AccF32,
     &VulkanGemmVariantFilter::allowCoopmat2,
     &VulkanGemmVariantFilter::allowAccF32,
     &VulkanGemmVariantFilter::allowCoopmat2AccF32,
     "coopmat2accf32",
     &VulkanTuner::coopmat2AccF32Bits},
    {GemmVariantTokenKind::Coopmat2AccF16,
     &VulkanGemmVariantFilter::allowCoopmat2,
     &VulkanGemmVariantFilter::allowAccF16,
     &VulkanGemmVariantFilter::allowCoopmat2AccF16,
     "coopmat2accf16",
     &VulkanTuner::coopmat2AccF16Bits},
    {GemmVariantTokenKind::Coopmat1AccF32,
     &VulkanGemmVariantFilter::allowCoopmat1,
     &VulkanGemmVariantFilter::allowAccF32,
     &VulkanGemmVariantFilter::allowCoopmat1AccF32,
     "coopmat1accf32",
     &VulkanTuner::coopmat1AccF32Bits},
    {GemmVariantTokenKind::Coopmat1AccF16,
     &VulkanGemmVariantFilter::allowCoopmat1,
     &VulkanGemmVariantFilter::allowAccF16,
     &VulkanGemmVariantFilter::allowCoopmat1AccF16,
     "coopmat1accf16",
     &VulkanTuner::coopmat1AccF16Bits},
    {GemmVariantTokenKind::Dot2AccF32,
     &VulkanGemmVariantFilter::allowDot2,
     &VulkanGemmVariantFilter::allowAccF32,
     &VulkanGemmVariantFilter::allowDot2AccF32,
     "dot2accf32",
     &VulkanTuner::dot2AccF32Bits},
    {GemmVariantTokenKind::Dot2AccF16,
     &VulkanGemmVariantFilter::allowDot2,
     &VulkanGemmVariantFilter::allowAccF16,
     &VulkanGemmVariantFilter::allowDot2AccF16,
     "dot2accf16",
     &VulkanTuner::dot2AccF16Bits},
  };

  const GemmVariantDescriptor* findGemmVariantDescriptor(GemmVariantTokenKind kind) {
    for(const auto& d: kGemmVariantDescriptors)
      if(d.kind == kind)
        return &d;
    return nullptr;
  }

  // True when the specific variant (kind) passes all three filter gates.
  bool gemmVariantAllowed(const VulkanGemmVariantFilter& filter, GemmVariantTokenKind kind) {
    const GemmVariantDescriptor* d = findGemmVariantDescriptor(kind);
    return d != nullptr && filter.*(d->familyField) && filter.*(d->accField) && filter.*(d->specificField);
  }

  // One row per accelerator family (coopmat1/coopmat2/dot2): the family gate
  // plus the two specific variants (FP32/FP16 accumulation) that hang off it.
  // accF16TogglesWithFamily is false for Dot2: a generic Dot2 request retains
  // the default FP32 accumulator, and the FP16 accumulator is opt-in only (the
  // explicit accf16/dot2accf16 token) — a family disable still clears it.
  struct GemmFamilyDescriptor {
    GemmVariantTokenKind kind;
    bool VulkanGemmVariantFilter::* familyField;
    const GemmVariantDescriptor* accF32Variant;
    const GemmVariantDescriptor* accF16Variant;
    bool accF16TogglesWithFamily;
  };

  const GemmFamilyDescriptor kGemmFamilyDescriptors[] = {
    {GemmVariantTokenKind::Coopmat2,
     &VulkanGemmVariantFilter::allowCoopmat2,
     &kGemmVariantDescriptors[0],
     &kGemmVariantDescriptors[1],
     true},
    {GemmVariantTokenKind::Coopmat1,
     &VulkanGemmVariantFilter::allowCoopmat1,
     &kGemmVariantDescriptors[2],
     &kGemmVariantDescriptors[3],
     true},
    {GemmVariantTokenKind::Dot2,
     &VulkanGemmVariantFilter::allowDot2,
     &kGemmVariantDescriptors[4],
     &kGemmVariantDescriptors[5],
     false},
  };

  void setGemmFamily(VulkanGemmVariantFilter& filter, GemmVariantTokenKind kind, bool enabled) {
    if(kind == GemmVariantTokenKind::Coopmat) {
      setGemmFamily(filter, GemmVariantTokenKind::Coopmat1, enabled);
      setGemmFamily(filter, GemmVariantTokenKind::Coopmat2, enabled);
      return;
    }
    for(const auto& [fdKind, familyField, accF32Variant, accF16Variant, accF16TogglesWithFamily]:
        kGemmFamilyDescriptors) {
      if(fdKind == kind) {
        filter.*(familyField) = enabled;
        filter.*(accF32Variant->specificField) = enabled;
        if(accF16TogglesWithFamily)
          filter.*(accF16Variant->specificField) = enabled;
        else if(!enabled)
          filter.*(accF16Variant->specificField) = false;
        return;
      }
    }
  }

  void setGemmAccumulator(VulkanGemmVariantFilter& filter, GemmVariantTokenKind kind, bool enabled) {
    if(kind == GemmVariantTokenKind::AccF16) {
      if(enabled) {
        filter.allowCoopmat2 = true;
        filter.allowCoopmat1 = true;
        filter.allowDot2 = true;
      }
      filter.allowAccF16 = enabled;
      filter.allowCoopmat2AccF16 = enabled;
      filter.allowCoopmat1AccF16 = enabled;
      filter.allowDot2AccF16 = enabled;
    } else if(kind == GemmVariantTokenKind::AccF32) {
      if(enabled) {
        filter.allowCoopmat2 = true;
        filter.allowCoopmat1 = true;
        filter.allowDot2 = true;
      }
      filter.allowAccF32 = enabled;
      filter.allowCoopmat2AccF32 = enabled;
      filter.allowCoopmat1AccF32 = enabled;
      filter.allowDot2AccF32 = enabled;
    }
  }

  void setAllGemmVariantMasks(VulkanGemmVariantFilter& filter, bool enabled) {
    for(const auto& d: kGemmVariantDescriptors)
      filter.*(d.specificField) = enabled;
  }

  void setSpecificGemmVariant(VulkanGemmVariantFilter& filter, GemmVariantTokenKind kind, bool enabled) {
    const GemmVariantDescriptor* d = findGemmVariantDescriptor(kind);
    if(d == nullptr)
      return;
    if(enabled) {
      filter.*(d->familyField) = true;
      filter.*(d->accField) = true;
    }
    filter.*(d->specificField) = enabled;
  }

  string gemmVariantFilterSummary(const VulkanGemmVariantFilter& filter) {
    vector<string> variants;
    for(const auto& d: kGemmVariantDescriptors)
      if(gemmVariantAllowed(filter, d.kind))
        variants.push_back(d.token);
    if(variants.empty())
      variants.push_back("tiled-only");

    return string("variants [") + Global::concat(variants, ", ") + "]";
  }

  void logVulkanEnvMessage(Logger* logger, const string& message) {
    if(logger != nullptr)
      logger->write(message);
    else
      cerr << message << endl;
  }

  string validGemmVariantFilterTokens() {
    return "coopmat, coopmat2, coopmat2accf32, coopmat2accf16, coopmat1, coopmat1accf32, coopmat1accf16, dot2, "
           "dot2accf32, dot2accf16, accf16, accf32";
  }

  VulkanGemmVariantFilter getVulkanGemmVariantFilter(Logger* logger) {
    VulkanGemmVariantFilter filter;

    string_view env = VulkanHelpers::getenvStringView("KATAGO_VULKAN_ACCEL_VARIANTS");
    vector<ParsedGemmVariantToken> tokens;
    if(!env.empty()) {
      filter.configured = true;
      const vector<string_view> rawTokens = splitEnvTokens(env);
      tokens.reserve(rawTokens.size());
      bool sawPositiveVariantToken = false;
      for(string_view rawToken: rawTokens) {
        ParsedGemmVariantToken parsed = parseGemmVariantToken(rawToken);
        if(
          isSpecificGemmVariantToken(parsed.kind) || isGemmFamilyToken(parsed.kind) ||
          isGemmAccumulatorToken(parsed.kind)) {
          sawPositiveVariantToken = sawPositiveVariantToken || !parsed.disable;
        } else if(parsed.kind == GemmVariantTokenKind::Unknown) {
          logVulkanEnvMessage(
            logger,
            "Warning: ignoring unrecognized token in KATAGO_VULKAN_ACCEL_VARIANTS: " + string(parsed.original) +
              " (valid values, optionally prefixed with '-' to disable: " + validGemmVariantFilterTokens() + ")");
        }
        tokens.push_back(parsed);
      }
      if(sawPositiveVariantToken)
        setAllGemmVariantMasks(filter, false);
    }

    for(const ParsedGemmVariantToken& token: tokens) {
      switch(token.kind) {
        case GemmVariantTokenKind::Coopmat:
        case GemmVariantTokenKind::Coopmat2:
        case GemmVariantTokenKind::Coopmat1:
        case GemmVariantTokenKind::Dot2:
          setGemmFamily(filter, token.kind, !token.disable);
          break;
        case GemmVariantTokenKind::AccF16:
        case GemmVariantTokenKind::AccF32:
          setGemmAccumulator(filter, token.kind, !token.disable);
          break;
        case GemmVariantTokenKind::Coopmat2AccF32:
        case GemmVariantTokenKind::Coopmat2AccF16:
        case GemmVariantTokenKind::Coopmat1AccF32:
        case GemmVariantTokenKind::Coopmat1AccF16:
        case GemmVariantTokenKind::Dot2AccF32:
        case GemmVariantTokenKind::Dot2AccF16:
          setSpecificGemmVariant(filter, token.kind, !token.disable);
          break;
        case GemmVariantTokenKind::Unknown:
          break;
        default:
          break;
      }
    }

    if(filter.configured && logger != nullptr)
      logger->write(string("Vulkan: acceleration variant filter -> ") + gemmVariantFilterSummary(filter));

    return filter;
  }

  void applyVulkanGemmVariantFilter(VulkanDeviceInfo& info, const VulkanGemmVariantFilter& filter) {
    // The filter records which accelerator variants are disabled without lying
    // about the device's capabilities: supports* and the shape vectors stay the
    // truthful hardware/driver report, and every consumer ANDs this mask out.
    for(const auto& d: kGemmVariantDescriptors)
      if(!gemmVariantAllowed(filter, d.kind))
        info.disabledAccelVariantMask |= (*d.disabledBits)();
  }

}  // namespace

void VulkanHelpers::checkResult(VkResult result, const char* file, int line) {
  if(result == VK_SUCCESS)
    return;
  const string msg = string("Vulkan error ") + string(vkResultString(result)) + " at " + file + ":" + to_string(line);
  // VK_ERROR_DEVICE_LOST is unrecoverable: the driver has reset the device and
  // every later call on this VkDevice fails too. Throw a distinct type so the
  // tuner's per-candidate handlers can rethrow instead of recording it as "this
  // candidate is invalid" -- otherwise one lost device turns into thousands of
  // bogus candidate rejections and a tune file with everything marked unmeasured.
  if(result == VK_ERROR_DEVICE_LOST)
    throw VulkanHelpers::DeviceLostError(msg);
  throw StringError(msg);
}

namespace {
  mutex g_pipelineExecutableStatsMutex;
  unordered_map<VkDevice, bool> g_pipelineExecutableStatsDevices;
  mutex g_shaderRteF16Mutex;
  unordered_map<VkDevice, bool> g_shaderRteF16Devices;

  void registerShaderRteF16Device(VkDevice device, bool enabled) {
    lock_guard<mutex> lock(g_shaderRteF16Mutex);
    g_shaderRteF16Devices[device] = enabled;
  }

  void unregisterShaderRteF16Device(VkDevice device) {
    lock_guard<mutex> lock(g_shaderRteF16Mutex);
    g_shaderRteF16Devices.erase(device);
  }

  bool shaderRteF16Enabled(VkDevice device) {
    lock_guard<mutex> lock(g_shaderRteF16Mutex);
    auto iter = g_shaderRteF16Devices.find(device);
    return iter != g_shaderRteF16Devices.end() && iter->second;
  }

#if defined(VK_KHR_pipeline_executable_properties)
  string pipelineExecutableStatValueString(const VkPipelineExecutableStatisticKHR& stat) {
    ostringstream out;
    switch(stat.format) {
      case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
        out << (stat.value.b32 == VK_TRUE ? "true" : "false");
        break;
      case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
        out << stat.value.i64;
        break;
      case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
        out << stat.value.u64;
        break;
      case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
        out << stat.value.f64;
        break;
      default:
        out << "<unknown>";
        break;
    }
    return out.str();
  }
#endif
}  // namespace

static bool pipelineExecutableStatsRequested() {
  static const bool cached = []() {
    string_view v = VulkanHelpers::getenvStringView("KATAGO_VULKAN_PIPELINE_STATS");
    return !v.empty() && v[0] != '0';
  }();
  return cached;
}

static void registerPipelineExecutableStatsDevice(VkDevice device, bool enabled) {
  lock_guard<mutex> lock(g_pipelineExecutableStatsMutex);
  g_pipelineExecutableStatsDevices[device] = enabled;
}

static void unregisterPipelineExecutableStatsDevice(VkDevice device) {
  lock_guard<mutex> lock(g_pipelineExecutableStatsMutex);
  g_pipelineExecutableStatsDevices.erase(device);
}

bool VulkanHelpers::pipelineExecutableStatsEnabled(VkDevice device) {
  if(!pipelineExecutableStatsRequested())
    return false;
  lock_guard<mutex> lock(g_pipelineExecutableStatsMutex);
  auto iter = g_pipelineExecutableStatsDevices.find(device);
  return iter != g_pipelineExecutableStatsDevices.end() && iter->second;
}

void VulkanHelpers::dumpPipelineExecutableStats(VkDevice device, VkPipeline pipeline, string_view label) {
#if defined(VK_KHR_pipeline_executable_properties)
  if(!pipelineExecutableStatsEnabled(device) || pipeline == VK_NULL_HANDLE)
    return;

  auto getProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
    vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
  auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
    vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
  if(getProps == nullptr || getStats == nullptr) {
    std::cerr << "VulkanPipelineStats: VK_KHR_pipeline_executable_properties enabled but query functions are missing"
              << std::endl;
    return;
  }

  VkPipelineInfoKHR pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
  pipelineInfo.pipeline = pipeline;

  uint32_t executableCount = 0;
  VkResult result = getProps(device, &pipelineInfo, &executableCount, nullptr);
  if(result != VK_SUCCESS || executableCount == 0)
    return;

  vector<VkPipelineExecutablePropertiesKHR> props(executableCount);
  for(auto& p: props)
    p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
  result = getProps(device, &pipelineInfo, &executableCount, props.data());
  if(result != VK_SUCCESS)
    return;

  string_view safeLabel = label.empty() ? string_view("<unnamed>") : label;
  for(uint32_t executableIdx = 0; executableIdx < executableCount; executableIdx++) {
    const auto& prop = props[executableIdx];
    std::cerr << "VulkanPipelineStats: " << safeLabel << " executable#" << executableIdx << " name=\"" << prop.name
              << "\" subgroup_size=" << prop.subgroupSize << std::endl;

    VkPipelineExecutableInfoKHR executableInfo = {};
    executableInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
    executableInfo.pipeline = pipeline;
    executableInfo.executableIndex = executableIdx;

    uint32_t statCount = 0;
    result = getStats(device, &executableInfo, &statCount, nullptr);
    if(result != VK_SUCCESS || statCount == 0)
      continue;
    vector<VkPipelineExecutableStatisticKHR> stats(statCount);
    for(auto& s: stats)
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    result = getStats(device, &executableInfo, &statCount, stats.data());
    if(result != VK_SUCCESS)
      continue;

    for(uint32_t statIdx = 0; statIdx < statCount; statIdx++) {
      const auto& stat = stats[statIdx];
      std::cerr << "VulkanPipelineStats:   " << stat.name << "=" << pipelineExecutableStatValueString(stat)
                << std::endl;
    }
  }
#else
  (void)device;
  (void)pipeline;
  (void)label;
#endif
}

// Memory helpers

static uint32_t
findMemoryType(const VkPhysicalDeviceMemoryProperties& memProps, uint32_t typeBits, VkMemoryPropertyFlags required) {
  for(uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & required) == required)
      return i;
  }
  throw StringError("Vulkan: no suitable memory type found");
}

VulkanBuffer VulkanHelpers::allocateBuffer(
  VkDevice device,
  const VkPhysicalDeviceMemoryProperties& memProps,
  VkDeviceSize size,
  VkBufferUsageFlags usage,
  VkMemoryPropertyFlags memFlags) {
  VkBufferCreateInfo bufInfo = {};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer;
  VK_CHECK(vkCreateBuffer(device, &bufInfo, nullptr, &buffer));

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(memProps, memReqs.memoryTypeBits, memFlags);

  VkDeviceMemory memory;
  VkResult allocResult = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
  if(allocResult != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    VK_CHECK(allocResult);
  }

  VkResult bindResult = vkBindBufferMemory(device, buffer, memory, 0);
  if(bindResult != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    VK_CHECK(bindResult);
  }

  return VulkanBuffer(buffer, memory, size, device);
}

static VulkanBuffer allocateAndUploadBuffer(
  VkDevice device,
  VkQueue queue,
  mutex& queueMutex,
  VkCommandPool commandPool,
  const VkPhysicalDeviceMemoryProperties& memProps,
  const void* data,
  VkDeviceSize size) {
  // Staging buffer (host-visible)
  VulkanBuffer staging = VulkanHelpers::allocateBuffer(
    device,
    memProps,
    size,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  void* mapped;
  VK_CHECK(vkMapMemory(device, staging.memory, 0, size, 0, &mapped));
  memcpy(mapped, data, static_cast<size_t>(size));
  vkUnmapMemory(device, staging.memory);

  // Device-local buffer (weights are always consumed as storage buffers)
  VulkanBuffer devBuf = VulkanHelpers::allocateBuffer(
    device,
    memProps,
    size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Transient command buffer for the copy
  VkCommandBufferAllocateInfo cbAllocInfo = {};
  cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAllocInfo.commandPool = commandPool;
  cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAllocInfo.commandBufferCount = 1;

  VkCommandBuffer cb;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAllocInfo, &cb));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cb, &beginInfo));

  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cb, staging.buffer, devBuf.buffer, 1, &region);

  VK_CHECK(vkEndCommandBuffer(cb));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cb;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

  {
    lock_guard<mutex> lock(queueMutex);
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
  }

  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, commandPool, 1, &cb);
  // staging destroyed here (RAII)

  return devBuf;
}

// Pipeline barriers

namespace {
  // The five VulkanHelpers::*Barrier entry points below differ only in their
  // access masks and stage masks; they share this single VkBufferMemoryBarrier
  // construction + vkCmdPipelineBarrier call.
  void bufferMemoryBarrier(
    VkCommandBuffer cmd,
    VkBuffer buf,
    VkPipelineStageFlags srcStage,
    VkAccessFlags srcAccess,
    VkPipelineStageFlags dstStage,
    VkAccessFlags dstAccess) {
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buf;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
  }
}  // namespace

void VulkanHelpers::cmdComputeBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  bufferMemoryBarrier(
    cmd,
    buf,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void VulkanHelpers::cmdComputeWARBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  // Execution-only ordering barrier on a named buffer.
  // srcStageMask=COMPUTE and dstStageMask=COMPUTE provide the execution dependency:
  // all compute dispatched before this barrier must finish before any compute
  // after it starts. This prevents the GPU scheduling a new writer to this buffer
  // before the current user finishes. Zero access masks mean no cache flush is
  // requested — for a scratch slot that will be completely overwritten by the
  // next writer, coherency is not needed, only ordering.
  bufferMemoryBarrier(cmd, buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);
}

void VulkanHelpers::cmdTransferToComputeBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  bufferMemoryBarrier(
    cmd,
    buf,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT);
}

void VulkanHelpers::cmdHostToComputeBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  bufferMemoryBarrier(
    cmd,
    buf,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_ACCESS_HOST_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT);
}

void VulkanHelpers::cmdComputeToTransferBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  bufferMemoryBarrier(
    cmd,
    buf,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_TRANSFER_READ_BIT);
}

// Shader module helper

VkShaderModule VulkanHelpers::createShaderModule(VkDevice device, const uint32_t* spirvData, size_t spirvWords) {
  const uint32_t* moduleData = spirvData;
  size_t moduleWords = spirvWords;
  if(shaderRteF16Enabled(device)) {
    const uint32_t* rteData = nullptr;
    size_t rteWords = 0;
    if(VulkanShaders::getRteVariant(spirvData, spirvWords, rteData, rteWords)) {
      moduleData = rteData;
      moduleWords = rteWords;
    }
  }

  VkShaderModuleCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = moduleWords * sizeof(uint32_t);
  info.pCode = moduleData;
  VkShaderModule module;
  VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
  return module;
}

// Device probing
namespace {

  // True when the reported apiVersion includes core support for the given
  // Vulkan (major, minor) version. Several feature gates below read the same
  // VK_VERSION_MAJOR/MINOR check against different API versions.
  bool apiHasCoreVersion(uint32_t apiVersion, int major, int minor) {
    return VK_VERSION_MAJOR(apiVersion) > major ||
           (VK_VERSION_MAJOR(apiVersion) == major && VK_VERSION_MINOR(apiVersion) >= minor);
  }

  // Shared context for probing a single physical device. The probeDevice*
  // helpers accumulate into the same VulkanDeviceInfo and several build on
  // earlier results (probeCoopmatBase gates the coopmat1/2 probes), so their
  // call order in VulkanDeviceInfo::getAllDeviceInfosOnSystem is significant.
  struct DeviceProbeContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceProperties2 getProps2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 getFeatures2 = nullptr;
    PFN_GetCoopmatProps getCoopmatProps = nullptr;
    vector<VkExtensionProperties> exts;

    bool hasExt(const char* name) const {
      for(const auto& e: exts)
        if(strcmp(e.extensionName, name) == 0)
          return true;
      return false;
    }
  };

  void probeDeviceProperties(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    if(ctx.getProps2 != nullptr) {
      VkPhysicalDeviceSubgroupProperties subgroupProps = {};
      subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
      VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeProps = {};
      subgroupSizeProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
      subgroupSizeProps.pNext = nullptr;
      subgroupProps.pNext = &subgroupSizeProps;
#else
      subgroupProps.pNext = nullptr;
#endif
      VkPhysicalDeviceProperties2 props2 = {};
      props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props2.pNext = &subgroupProps;
      ctx.getProps2(ctx.pd, &props2);
      info.properties = props2.properties;
      info.subgroupSize = subgroupProps.subgroupSize;
      const VkSubgroupFeatureFlags subgroupOpsRequired =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
      info.supportsSubgroupShuffleCompute =
        subgroupProps.subgroupSize > 0 && (subgroupProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroupProps.supportedOperations & subgroupOpsRequired) == subgroupOpsRequired;
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
      info.minSubgroupSize = subgroupSizeProps.minSubgroupSize;
      info.maxSubgroupSize = subgroupSizeProps.maxSubgroupSize;
      info.supportsRequiredSubgroupSizeCompute =
        (subgroupSizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
#endif
    } else {
      vkGetPhysicalDeviceProperties(ctx.pd, &info.properties);
    }
    vkGetPhysicalDeviceMemoryProperties(ctx.pd, &info.memoryProperties);
  }

  void probeDeviceFeatures(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    if(ctx.getFeatures2 == nullptr)
      return;
    VkPhysicalDeviceFeatures2 feats2 = {};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroupFeat = {};
    subgroupFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroupFeat.pNext = nullptr;
    feats2.pNext = &subgroupFeat;
#else
    feats2.pNext = nullptr;
#endif
    ctx.getFeatures2(ctx.pd, &feats2);
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
    info.supportsSubgroupSizeControl = subgroupFeat.subgroupSizeControl == VK_TRUE;
    info.supportsComputeFullSubgroups = subgroupFeat.computeFullSubgroups == VK_TRUE;
#endif
  }

  // Returns false when the device has no compute queue family (caller skips it).
  bool probeQueueFamilies(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.pd, &qfCount, nullptr);
    vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.pd, &qfCount, qfProps.data());

    info.computeQueueFamilyIdx = -1;
    info.transferQueueFamilyIdx = -1;
    info.computeQueueCount = 0;
    for(uint32_t q = 0; q < qfCount; q++) {
      if(qfProps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        info.computeQueueFamilyIdx = static_cast<int>(q);
        info.computeQueueCount = static_cast<int>(qfProps[q].queueCount);
        break;
      }
    }
    if(info.computeQueueFamilyIdx < 0)
      return false;
    for(uint32_t q = 0; q < qfCount; q++) {
      if((qfProps[q].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qfProps[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        info.transferQueueFamilyIdx = static_cast<int>(q);
        break;
      }
    }
    if(info.transferQueueFamilyIdx < 0)
      info.transferQueueFamilyIdx = info.computeQueueFamilyIdx;
    return true;
  }

  void probeDeviceExtensions(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(ctx.pd, nullptr, &extCount, nullptr);
    ctx.exts.resize(extCount);
    vkEnumerateDeviceExtensionProperties(ctx.pd, nullptr, &extCount, ctx.exts.data());

    info.supportsPushDescriptors = ctx.hasExt(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    info.hasSubgroupSizeControlExtension = ctx.hasExt(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
  }

  void probeShaderFloatControls(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    const bool apiHasCoreShaderFloatControls = apiHasCoreVersion(info.properties.apiVersion, 1, 2);
    if((apiHasCoreShaderFloatControls || ctx.hasExt(kShaderFloatControlsExtensionName)) && ctx.getProps2 != nullptr) {
      FloatControlsProperties floatControlsProps = {};
      floatControlsProps.sType = kSTypeFloatControlsProperties;
      floatControlsProps.pNext = nullptr;
      VkPhysicalDeviceProperties2 props2FloatControls = {};
      props2FloatControls.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props2FloatControls.pNext = &floatControlsProps;
      ctx.getProps2(ctx.pd, &props2FloatControls);
      info.supportsShaderRoundingModeRTEFloat16 = floatControlsProps.shaderRoundingModeRTEFloat16 == VK_TRUE;
      info.shaderFloatControlsNeedsExt = !apiHasCoreShaderFloatControls;
    }
  }

#if defined(VK_KHR_pipeline_executable_properties)
  void probePipelineExecutableProperties(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    if(ctx.hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) && ctx.getFeatures2 != nullptr) {
      VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExecutableFeats = {};
      pipelineExecutableFeats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
      VkPhysicalDeviceFeatures2 feats2PipelineExecutable = {};
      feats2PipelineExecutable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2PipelineExecutable.pNext = &pipelineExecutableFeats;
      ctx.getFeatures2(ctx.pd, &feats2PipelineExecutable);
      info.supportsPipelineExecutableProperties = pipelineExecutableFeats.pipelineExecutableInfo == VK_TRUE;
    }
  }
#endif

  void probe16BitStorage(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    const bool apiHasCore16BitStorage = apiHasCoreVersion(info.properties.apiVersion, 1, 1);
    if((apiHasCore16BitStorage || ctx.hasExt(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) && ctx.getFeatures2 != nullptr) {
      VkPhysicalDevice16BitStorageFeatures storageFeats = {};
      storageFeats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
      storageFeats.pNext = nullptr;
      VkPhysicalDeviceFeatures2 feats2storage = {};
      feats2storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2storage.pNext = &storageFeats;
      ctx.getFeatures2(ctx.pd, &feats2storage);
      info.supportsFP16Storage = storageFeats.storageBuffer16BitAccess == VK_TRUE;
    }
  }

#if defined(KATAGO_HAS_SHADER_FLOAT16_INT8_HEADERS)
  void probeShaderFloat16Int8(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    const bool apiHasCoreShaderFloat16Int8 = apiHasCoreVersion(info.properties.apiVersion, 1, 2);
    if(
      (apiHasCoreShaderFloat16Int8 || ctx.hasExt(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) &&
      ctx.getFeatures2 != nullptr) {
      VkPhysicalDeviceShaderFloat16Int8Features f16Feats = {};
      f16Feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
      f16Feats.pNext = nullptr;
      VkPhysicalDeviceFeatures2 feats2f16 = {};
      feats2f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2f16.pNext = &f16Feats;
      ctx.getFeatures2(ctx.pd, &feats2f16);
      info.supportsFP16Compute = f16Feats.shaderFloat16 == VK_TRUE;
    }
  }
#endif

  // VK_VALVE_shader_mixed_float_dot_product — provides OpFDot2MixAcc32/16VALVE.
  // The Vulkan extension is not yet in system headers; the struct definition
  // lives in the file-scope anonymous namespace at the top of this file.
  void probeDot2(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    if(ctx.hasExt("VK_VALVE_shader_mixed_float_dot_product") && ctx.getFeatures2 != nullptr) {
      VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local dot2Feats = {};
      dot2Feats.sType = kVkStructureTypeShaderMixedFloatDotProductFeaturesVALVE;
      dot2Feats.pNext = nullptr;
      VkPhysicalDeviceFeatures2 feats2dot2 = {};
      feats2dot2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2dot2.pNext = &dot2Feats;
      ctx.getFeatures2(ctx.pd, &feats2dot2);
      info.supportsDot2F16 =
        dot2Feats.shaderMixedFloatDotProductFloat16AccFloat32 == VK_TRUE && info.supportsFP16Compute;
      info.supportsDot2F16AccF16 =
        dot2Feats.shaderMixedFloatDotProductFloat16AccFloat16 == VK_TRUE && info.supportsFP16Compute;
    }
  }

  // When the device has neither core-1.3 subgroup-size control nor the
  // VK_EXT_subgroup_size_control extension, force the size-control feature
  // fields off (they are queried only when at least one of those paths exists).
  void probeSubgroupSizeControlReset(VulkanDeviceInfo& info) {
    const bool apiHasCoreSubgroupSizeControl = apiHasCoreVersion(info.properties.apiVersion, 1, 3);
    if(!apiHasCoreSubgroupSizeControl && !info.hasSubgroupSizeControlExtension) {
      info.supportsSubgroupSizeControl = false;
      info.supportsComputeFullSubgroups = false;
      info.supportsRequiredSubgroupSizeCompute = false;
      info.minSubgroupSize = 0;
      info.maxSubgroupSize = 0;
    }
  }

  // VK_KHR_cooperative_matrix base. Coopmat1 and coopmat2 both require the
  // common KHR feature, FP16 base, Vulkan memory model, and compute-stage
  // support, but only coopmat1 inherits subgroup-size-control policy.
  void probeCoopmatBase(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    const bool coopmatFp16Base = info.supportsFP16Storage && info.supportsFP16Compute;
    const bool apiHasCoreMemModel = apiHasCoreVersion(info.properties.apiVersion, 1, 2);
    const bool hasMemModelFeaturePath = apiHasCoreMemModel || ctx.hasExt(kVulkanMemoryModelExtensionName);
    bool computeStageSupported = false;
    if(ctx.hasExt(kCoopmatExtensionName) && ctx.getFeatures2 != nullptr && coopmatFp16Base && hasMemModelFeaturePath) {
      CoopmatFeaturesKHR cmFeats = {};
      cmFeats.sType = kSTypeCoopmatFeatures;
      cmFeats.pNext = nullptr;
      // The coopmat shaders declare OpCapability VulkanMemoryModel (from
      // GL_KHR_memory_scope_semantics), so the vulkanMemoryModel feature is a
      // hard requirement — query it in the same features2 chain.
      VulkanMemoryModelFeatures memModelFeats = {};
      memModelFeats.sType = kSTypeVulkanMemoryModelFeatures;
      memModelFeats.pNext = &cmFeats;
      VkPhysicalDeviceFeatures2 feats2cm = {};
      feats2cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2cm.pNext = &memModelFeats;
      ctx.getFeatures2(ctx.pd, &feats2cm);
      info.supportsVulkanMemoryModel = (memModelFeats.vulkanMemoryModel == VK_TRUE);

      ctx.getCoopmatProps = reinterpret_cast<PFN_GetCoopmatProps>(
        vkGetInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
      // Coopmat must be usable from a compute shader specifically: a device may
      // advertise shapes only for other stages. The stage mask lives on the
      // physical-device coopmat properties (queried via GetProperties2), which
      // is distinct from the per-shape enumeration below.
      if(ctx.getProps2 != nullptr) {
        CoopmatDevicePropertiesKHR cmDeviceProps = {};
        cmDeviceProps.sType = kSTypeCoopmatDeviceProperties;
        cmDeviceProps.pNext = nullptr;
        VkPhysicalDeviceProperties2 props2cm = {};
        props2cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2cm.pNext = &cmDeviceProps;
        ctx.getProps2(ctx.pd, &props2cm);
        computeStageSupported = (cmDeviceProps.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
      }
      info.supportsKhrCoopmatBase =
        cmFeats.cooperativeMatrix == VK_TRUE && info.supportsVulkanMemoryModel && computeStageSupported;
      if(info.supportsKhrCoopmatBase) {
        info.coopmatNeedsMemModelExt = !apiHasCoreMemModel;
        if(info.coopmatNeedsMemModelExt && !ctx.hasExt(kVulkanMemoryModelExtensionName))
          info.supportsKhrCoopmatBase = false;
      }
    }
  }

#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
  // VK_KHR_cooperative_matrix — the coopmat1 GEMM paths. Eligible only on the
  // common KHR base, with compute-capable subgroup-scope f16xf16 shapes, and a
  // subgroup-size guarantee (coopmat pipelines are built with a required
  // subgroup size + full subgroups). FP32-accum and FP16-accum shapes are
  // recorded separately because devices may support one but not the other.
  void probeCoopmat1(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    if(info.supportsKhrCoopmatBase && info.canRequireComputeSubgroupSize(info.subgroupSize)) {
      if(ctx.getCoopmatProps != nullptr) {
        uint32_t propCount = 0;
        if(ctx.getCoopmatProps(ctx.pd, &propCount, nullptr) == VK_SUCCESS && propCount > 0) {
          vector<CoopmatPropertiesKHR> props(propCount);
          for(auto& p: props) {
            p.sType = kSTypeCoopmatProperties;
            p.pNext = nullptr;
          }
          if(ctx.getCoopmatProps(ctx.pd, &propCount, props.data()) == VK_SUCCESS) {
            for(const CoopmatPropertiesKHR& p: props) {
              const bool baseShape = p.scope == kScopeSubgroup && p.AType == kComponentFloat16 &&
                                     p.BType == kComponentFloat16 && p.MSize > 0 && p.NSize > 0 && p.KSize > 0;
              const bool matchesF32Acc = baseShape && p.CType == kComponentFloat32 && p.ResultType == kComponentFloat32;
              const bool matchesF16Acc = baseShape && p.CType == kComponentFloat16 && p.ResultType == kComponentFloat16;
              if(matchesF32Acc) {
                CoopmatShape shape{p.MSize, p.NSize, p.KSize};
                bool dup = false;
                for(const auto& s: info.coopmatShapes)
                  if(s == shape)
                    dup = true;
                if(!dup)
                  info.coopmatShapes.push_back(shape);
              }
              if(matchesF16Acc) {
                CoopmatShape shape{p.MSize, p.NSize, p.KSize};
                bool dup = false;
                for(const auto& s: info.coopmatAccF16Shapes)
                  if(s == shape)
                    dup = true;
                if(!dup)
                  info.coopmatAccF16Shapes.push_back(shape);
              }
            }
          }
        }
      }
      info.supportsCoopmat1F16 = !info.coopmatShapes.empty();
      info.supportsCoopmat1F16AccF16 = !info.coopmatAccF16Shapes.empty();
    }

#if defined(KATAGO_VULKAN_HAS_COOPMAT_MAINTENANCE1_SHADERS)
    // The experimental attention path needs all three operations. Do not infer
    // them from the extension name: maintenance1 advertises these feature bits
    // independently, and a device may expose only a subset.
    // This is deliberately a VkGetPhysicalDeviceFeatures2 query rather than
    // an extension-name heuristic. Drivers may advertise maintenance1 while
    // exposing only a subset of its independently enabled operations.
    if(
      info.supportsKhrCoopmatBase && info.supportsCoopmat1F16 && ctx.hasExt(kCoopmatMaintenance1ExtensionName) &&
      ctx.getFeatures2 != nullptr) {
      CoopmatMaintenance1FeaturesEXT maint1Feats = {};
      maint1Feats.sType = kSTypeCoopmatMaintenance1Features;
      VkPhysicalDeviceFeatures2 feats2Maint1 = {};
      feats2Maint1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2Maint1.pNext = &maint1Feats;
      ctx.getFeatures2(ctx.pd, &feats2Maint1);
      info.supportsCoopmatMaintenance1 = maint1Feats.cooperativeMatrixReductions == VK_TRUE &&
                                         maint1Feats.cooperativeMatrixConversions == VK_TRUE &&
                                         maint1Feats.cooperativeMatrixPerElementOperations == VK_TRUE;
    }
#endif
  }
#endif

#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
  // VK_NV_cooperative_matrix2 — workgroup-scope flexible-dimension GEMM path.
  void probeCoopmat2(DeviceProbeContext& ctx, VulkanDeviceInfo& info) {
    const bool apiHasSpirv16 = apiHasCoreVersion(info.properties.apiVersion, 1, 3);
    if(
      info.supportsKhrCoopmatBase && ctx.hasExt(kCoopmat2ExtensionName) && ctx.getFeatures2 != nullptr &&
      ctx.getProps2 != nullptr && apiHasSpirv16) {
      Coopmat2FeaturesNV cm2Feats = {};
      cm2Feats.sType = kSTypeCoopmat2Features;
      cm2Feats.pNext = nullptr;
      VkPhysicalDeviceFeatures2 feats2cm2 = {};
      feats2cm2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      feats2cm2.pNext = &cm2Feats;
      ctx.getFeatures2(ctx.pd, &feats2cm2);

      Coopmat2DevicePropertiesNV cm2Props = {};
      cm2Props.sType = kSTypeCoopmat2DeviceProperties;
      cm2Props.pNext = nullptr;
      VkPhysicalDeviceProperties2 props2cm2 = {};
      props2cm2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props2cm2.pNext = &cm2Props;
      ctx.getProps2(ctx.pd, &props2cm2);
      info.coopmat2MaxWorkgroupSize = cm2Props.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
      info.coopmat2MaxFlexDimension = cm2Props.cooperativeMatrixFlexibleDimensionsMaxDimension;
      info.coopmat2ReservedSharedBytes = cm2Props.cooperativeMatrixWorkgroupScopeReservedSharedMemory;
      // Record which optional NV coopmat2 features the driver reports. Only
      // tensor addressing is required for the GEMM paths; the attention shader
      // additionally needs reductions, conversions, per-element operations, and
      // block loads (folded into supportsCoopmat2Attention below).
      info.supportsCoopmat2Reductions = cm2Feats.cooperativeMatrixReductions == VK_TRUE;
      info.supportsCoopmat2Conversions = cm2Feats.cooperativeMatrixConversions == VK_TRUE;
      info.supportsCoopmat2PerElementOps = cm2Feats.cooperativeMatrixPerElementOperations == VK_TRUE;
      info.supportsCoopmat2BlockLoads = cm2Feats.cooperativeMatrixBlockLoads == VK_TRUE;

      auto getCoopmat2FlexProps = reinterpret_cast<PFN_GetCoopmat2FlexProps>(
        vkGetInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV"));
      const bool requiredFeatures =
        coopmat2RequiredFeaturesSupported(cm2Feats) && cm2Props.cooperativeMatrixWorkgroupScopeReservedSharedMemory <=
                                                         info.properties.limits.maxComputeSharedMemorySize;
      if(requiredFeatures && getCoopmat2FlexProps != nullptr) {
        uint32_t propCount = 0;
        VkResult res = getCoopmat2FlexProps(ctx.pd, &propCount, nullptr);
        if((res == VK_SUCCESS || res == VK_INCOMPLETE) && propCount > 0) {
          vector<Coopmat2FlexPropertiesNV> props(propCount);
          for(auto& p: props) {
            p.sType = kSTypeCoopmat2FlexProperties;
            p.pNext = nullptr;
          }
          res = getCoopmat2FlexProps(ctx.pd, &propCount, props.data());
          if(res == VK_SUCCESS || res == VK_INCOMPLETE) {
            const size_t numPropsRead = std::min<size_t>(propCount, props.size());
            for(size_t propIdx = 0; propIdx < numPropsRead; propIdx++) {
              const Coopmat2FlexPropertiesNV& p = props[propIdx];
              const bool positiveShape =
                p.MGranularity > 0 && p.NGranularity > 0 && p.KGranularity > 0 && p.workgroupInvocations > 0;
              const bool workgroupWithinLimit =
                cm2Props.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize == 0 ||
                p.workgroupInvocations <= cm2Props.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
              const bool baseShape = p.scope == kScopeWorkgroup && p.saturatingAccumulation == VK_FALSE &&
                                     p.AType == kComponentFloat16 && p.BType == kComponentFloat16 && positiveShape &&
                                     workgroupWithinLimit;
              const bool acceptedF32Acc =
                baseShape && p.CType == kComponentFloat32 && p.ResultType == kComponentFloat32;
              const bool acceptedF16Acc =
                baseShape && p.CType == kComponentFloat16 && p.ResultType == kComponentFloat16;
              if(acceptedF32Acc) {
                Coopmat2FlexShape shape{p.MGranularity, p.NGranularity, p.KGranularity, p.workgroupInvocations};
                bool dup = false;
                for(const auto& s: info.coopmat2FlexShapes)
                  if(s == shape)
                    dup = true;
                if(!dup)
                  info.coopmat2FlexShapes.push_back(shape);
              }
              if(acceptedF16Acc) {
                Coopmat2FlexShape shape{p.MGranularity, p.NGranularity, p.KGranularity, p.workgroupInvocations};
                bool dup = false;
                for(const auto& s: info.coopmat2AccF16FlexShapes)
                  if(s == shape)
                    dup = true;
                if(!dup)
                  info.coopmat2AccF16FlexShapes.push_back(shape);
              }
            }
          }
        }
      }
      info.supportsCoopmat2F16 = !info.coopmat2FlexShapes.empty();
      info.supportsCoopmat2F16AccF16 = !info.coopmat2AccF16FlexShapes.empty();
      // The coopmat2 attention shader needs the full optional feature set on top
      // of the shared coopmat2 base. Coopmat1 remains the attention fallback on
      // devices missing any of these.
      info.supportsCoopmat2Attention = info.supportsCoopmat2F16 && info.supportsCoopmat2Reductions &&
                                       info.supportsCoopmat2Conversions && info.supportsCoopmat2PerElementOps &&
                                       info.supportsCoopmat2BlockLoads;
    }
  }
#endif

}  // namespace

// Device info enumeration

vector<VulkanDeviceInfo> VulkanDeviceInfo::getAllDeviceInfosOnSystem(VkInstance instance, Logger* logger) {
  uint32_t deviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
  if(deviceCount == 0)
    return {};

  vector<VkPhysicalDevice> physDevices(deviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, physDevices.data()));

  vector<VulkanDeviceInfo> infos;
  infos.reserve(deviceCount);
  const VulkanGemmVariantFilter gemmVariantFilter = getVulkanGemmVariantFilter(logger);

  for(uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDevice pd = physDevices[i];
    VulkanDeviceInfo info = {};
    info.gpuIdx = static_cast<int>(i);
    info.physicalDevice = pd;

    DeviceProbeContext ctx;
    ctx.instance = instance;
    ctx.pd = pd;
    ctx.getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
    ctx.getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));

    probeDeviceProperties(ctx, info);
    probeDeviceFeatures(ctx, info);
    if(!probeQueueFamilies(ctx, info)) {
      if(logger)
        logger->write("Vulkan: device " + string(info.properties.deviceName) + " has no compute queue, skipping");
      continue;
    }
    probeDeviceExtensions(ctx, info);
    probeShaderFloatControls(ctx, info);
#if defined(VK_KHR_pipeline_executable_properties)
    probePipelineExecutableProperties(ctx, info);
#endif
    probe16BitStorage(ctx, info);
#if defined(KATAGO_HAS_SHADER_FLOAT16_INT8_HEADERS)
    probeShaderFloat16Int8(ctx, info);
#endif
    probeDot2(ctx, info);
    probeSubgroupSizeControlReset(info);
    probeCoopmatBase(ctx, info);
#if defined(KATAGO_VULKAN_HAS_COOPMAT_SHADERS)
    probeCoopmat1(ctx, info);
#endif
#if defined(KATAGO_VULKAN_HAS_COOPMAT2_SHADERS)
    probeCoopmat2(ctx, info);
#endif

    applyVulkanGemmVariantFilter(info, gemmVariantFilter);
    info.isIntegrated = (info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

    // Compute desirability score
    int score = 0;
    if(info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score += 1000;
    if(info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score += 100;
    if(info.supportsFP16Compute)
      score += 50;
    if(info.supportsFP16Storage)
      score += 10;
    if(info.supportsSubgroupShuffleCompute)
      score += 20;
    if(info.supportsCoopmat1F16)
      score += 40;
    if(info.supportsCoopmat1F16AccF16)
      score += 40;
    if(info.supportsCoopmat2F16)
      score += 60;
    if(info.supportsCoopmat2F16AccF16)
      score += 60;
    info.defaultDesirability = score;

    infos.push_back(info);
  }

  return infos;
}

// InitializedVulkanDevice destructor

InitializedVulkanDevice::~InitializedVulkanDevice() {
  if(device != VK_NULL_HANDLE) {
    if(transferCommandPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(device, transferCommandPool, nullptr);
    if(commandPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(device, commandPool, nullptr);
    unregisterPipelineExecutableStatsDevice(device);
    unregisterShaderRteF16Device(device);
    vkDestroyDevice(device, nullptr);
  }
}

InitializedVulkanDevice::QueueContext InitializedVulkanDevice::pickComputeContext() const {
  if(computeQueues.empty())
    return QueueContext{computeQueue, const_cast<std::mutex*>(&queueMutex), commandPool};
  const uint32_t idx =
    nextComputeQueueIdx.fetch_add(1, std::memory_order_relaxed) % static_cast<uint32_t>(computeQueues.size());
  if(idx == 0)
    return QueueContext{computeQueues[idx], const_cast<std::mutex*>(&queueMutex), commandPool};
  return QueueContext{computeQueues[idx], extraComputeQueueMutexes[idx - 1].get(), commandPool};
}

InitializedVulkanDevice::QueueContext InitializedVulkanDevice::getTransferContext() const {
  if(transferQueue != VK_NULL_HANDLE && transferQueueMutex != nullptr && transferCommandPool != VK_NULL_HANDLE) {
    return QueueContext{transferQueue, transferQueueMutex, transferCommandPool};
  }
  return QueueContext{computeQueue, const_cast<std::mutex*>(&queueMutex), commandPool};
}

// VulkanDevicesContext

namespace {

  static string vulkanFeatureListString(const vector<string>& features) {
    return features.empty() ? "none" : Global::concat(features, ", ");
  }

  static string vulkanAccelerationFeatureSummary(const VulkanDeviceInfo& info) {
    vector<string> enabled;
    vector<string> disabled;
    vector<string> unsupported;

    const int64_t dm = info.disabledAccelVariantMask;
    auto labelWithCount = [](const string& prefix, size_t count, const string& singular, const string& plural) {
      return prefix + " " + to_string(count) + (count == 1 ? singular : plural);
    };
    auto addFeature = [&](
                        bool supported,
                        bool filterDisabled,
                        const string& enabledLabel,
                        const string& disabledLabel,
                        const string& unsupportedLabel) {
      if(supported && !filterDisabled)
        enabled.push_back(enabledLabel);
      else if(supported)
        disabled.push_back(disabledLabel);
      else
        unsupported.push_back(unsupportedLabel);
    };

    addFeature(
      info.supportsDot2F16, (dm & VulkanTuner::dot2AccF32Bits()) != 0, "Dot2AccF32", "Dot2AccF32", "Dot2AccF32");
    addFeature(
      info.supportsDot2F16AccF16, (dm & VulkanTuner::dot2AccF16Bits()) != 0, "Dot2AccF16", "Dot2AccF16", "Dot2AccF16");
    addFeature(
      info.supportsCoopmat1F16,
      (dm & VulkanTuner::coopmat1AccF32Bits()) != 0,
      labelWithCount("Coopmat_KHR (V1,", info.coopmatShapes.size(), " shape)", " shapes)"),
      labelWithCount("Coopmat_KHR (V1,", info.coopmatShapes.size(), " shape)", " shapes)"),
      "Coopmat_KHR (V1)");
    addFeature(
      info.supportsCoopmat1F16AccF16,
      (dm & VulkanTuner::coopmat1AccF16Bits()) != 0,
      labelWithCount("Coopmat_KHR_AccF16 (V1,", info.coopmatAccF16Shapes.size(), " shape)", " shapes)"),
      labelWithCount("Coopmat_KHR_AccF16 (V1,", info.coopmatAccF16Shapes.size(), " shape)", " shapes)"),
      "Coopmat_KHR_AccF16 (V1)");
    addFeature(
      info.supportsCoopmat2F16,
      (dm & VulkanTuner::coopmat2AccF32Bits()) != 0,
      labelWithCount("Coopmat_NV (V2,", info.coopmat2FlexShapes.size(), " flex shape)", " flex shapes)"),
      labelWithCount("Coopmat_NV (V2,", info.coopmat2FlexShapes.size(), " flex shape)", " flex shapes)"),
      "Coopmat_NV (V2)");
    addFeature(
      info.supportsCoopmat2F16AccF16,
      (dm & VulkanTuner::coopmat2AccF16Bits()) != 0,
      labelWithCount("Coopmat_NV_AccF16 (V2,", info.coopmat2AccF16FlexShapes.size(), " flex shape)", " flex shapes)"),
      labelWithCount("Coopmat_NV_AccF16 (V2,", info.coopmat2AccF16FlexShapes.size(), " flex shape)", " flex shapes)"),
      "Coopmat_NV_AccF16 (V2)");
    addFeature(
      info.supportsCoopmatMaintenance1,
      (dm & (VulkanTuner::TUNED_ATTN_MAINTENANCE1 | VulkanTuner::TUNED_ATTN_MAINTENANCE1_SPLITK)) != 0,
      "Coopmat_KHR_Maintenance (V1)",
      "Coopmat_KHR_Maintenance (V1)",
      "Coopmat_KHR_Maintenance (V1)");
    addFeature(
      info.supportsCoopmat2Attention,
      (dm & VulkanTuner::TUNED_ATTN_COOPMAT2) != 0,
      "Coopmat_NV_Attention (V2)",
      "Coopmat_NV_Attention (V2)",
      "Coopmat_NV_Attention (V2)");

    return string("enabled [") + vulkanFeatureListString(enabled) + "]; disabled [" +
           vulkanFeatureListString(disabled) + "]; unsupported [" + vulkanFeatureListString(unsupported) + "]";
  }

  // The set of device extensions to enable, plus the acceleration features that
  // actually end up enabled. The feature flags are used downstream (the device
  // create feature chain, the post-create info reflection, and RTE/pipeline-stats
  // registration), so they are computed here alongside the extension list.
  struct DeviceExtensionSet {
    vector<const char*> names;
    bool enableShaderRoundingModeRTEFloat16 = false;
    bool enableFP16Compute = false;
    bool enableDot2F16 = false;
    bool enableDot2F16AccF16 = false;
    bool enableCoopmat1F16 = false;
    bool enableCoopmat1F16AccF16 = false;
    bool enableCoopmat2F16 = false;
    bool enableCoopmat2F16AccF16 = false;
    bool enableKhrCoopmatBase = false;
    bool enableCoopmatMaintenance1 = false;
    bool enablePipelineExecutableStats = false;
  };

  DeviceExtensionSet buildDeviceExtensionSet(
    const VulkanDeviceInfo& info,
    bool requestFP16Storage,
    bool requestFP16Compute,
    Logger* logger) {
    DeviceExtensionSet set;
    auto& extensions = set.names;

    // Decide which optional extensions to enable
    if(info.supportsPushDescriptors)
      extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    const bool apiHasCore16BitStorage = apiHasCoreVersion(info.properties.apiVersion, 1, 1);
    if(requestFP16Storage && info.supportsFP16Storage && !apiHasCore16BitStorage)
      extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    set.enableShaderRoundingModeRTEFloat16 = requestFP16Storage && info.supportsFP16Storage &&
                                             info.supportsShaderRoundingModeRTEFloat16 &&
                                             VulkanShaders::hasRteVariants();
    if(set.enableShaderRoundingModeRTEFloat16 && info.shaderFloatControlsNeedsExt)
      extensions.push_back(kShaderFloatControlsExtensionName);
    set.enableFP16Compute =
      requestFP16Compute && requestFP16Storage && info.supportsFP16Storage && info.supportsFP16Compute;
    const bool apiHasCoreShaderFloat16Int8 = apiHasCoreVersion(info.properties.apiVersion, 1, 2);
    if(set.enableFP16Compute && !apiHasCoreShaderFloat16Int8)
      extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    const bool apiHasCoreSubgroupSizeControl = apiHasCoreVersion(info.properties.apiVersion, 1, 3);
    if(
      !apiHasCoreSubgroupSizeControl && (info.supportsSubgroupSizeControl || info.supportsComputeFullSubgroups) &&
      info.hasSubgroupSizeControlExtension)
      extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    // DOT2 requires FP16 storage and the shaderFloat16 feature. The shaders load
    // f16 storage data and declare the SPIR-V Float16 capability for f16 vectors.
    set.enableDot2F16 = info.supportsDot2F16 && set.enableFP16Compute;
    set.enableDot2F16AccF16 = info.supportsDot2F16AccF16 && set.enableFP16Compute;
    if(set.enableDot2F16 || set.enableDot2F16AccF16)
      extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
    // Coopmat1 requires the same FP16 base as DOT2, plus the KHR coopmat
    // extension and a device-reported usable subgroup-scope shape. Coopmat2
    // branches from the same KHR base but adds VK_NV_cooperative_matrix2.
    set.enableCoopmat1F16 = info.supportsCoopmat1F16 && set.enableFP16Compute;
    set.enableCoopmat1F16AccF16 = info.supportsCoopmat1F16AccF16 && set.enableFP16Compute;
    set.enableCoopmat2F16 = info.supportsCoopmat2F16 && set.enableFP16Compute;
    set.enableCoopmat2F16AccF16 = info.supportsCoopmat2F16AccF16 && set.enableFP16Compute;
    set.enableKhrCoopmatBase =
      (set.enableCoopmat1F16 || set.enableCoopmat1F16AccF16 || set.enableCoopmat2F16 || set.enableCoopmat2F16AccF16) &&
      info.supportsKhrCoopmatBase;
    // maintenance1 is an attention-only extension. It is meaningful only when
    // the base KHR matrix feature and FP16 path are enabled as well.
#if defined(KATAGO_VULKAN_HAS_COOPMAT_MAINTENANCE1_SHADERS)
    set.enableCoopmatMaintenance1 =
      info.supportsCoopmatMaintenance1 && set.enableCoopmat1F16 && set.enableKhrCoopmatBase;
#else
    set.enableCoopmatMaintenance1 = false;
#endif
    if(set.enableKhrCoopmatBase)
      extensions.push_back(kCoopmatExtensionName);
    if(set.enableCoopmatMaintenance1)
      extensions.push_back(kCoopmatMaintenance1ExtensionName);
    if(set.enableCoopmat2F16 || set.enableCoopmat2F16AccF16)
      extensions.push_back(kCoopmat2ExtensionName);
    // The coopmat shaders declare OpCapability VulkanMemoryModel; on API < 1.2
    // that requires the VK_KHR_vulkan_memory_model extension in addition to the
    // enabled feature (core from 1.2 onward, so no extension needed there).
    if(set.enableKhrCoopmatBase && info.coopmatNeedsMemModelExt)
      extensions.push_back(kVulkanMemoryModelExtensionName);
#if defined(VK_KHR_pipeline_executable_properties)
    set.enablePipelineExecutableStats = pipelineExecutableStatsRequested() && info.supportsPipelineExecutableProperties;
    if(set.enablePipelineExecutableStats)
      extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
#else
    set.enablePipelineExecutableStats = false;
#endif
    if(pipelineExecutableStatsRequested() && !set.enablePipelineExecutableStats && logger != nullptr)
      logger->write(
        string("Vulkan: ") + info.properties.deviceName +
        " does not support VK_KHR_pipeline_executable_properties; KATAGO_VULKAN_PIPELINE_STATS disabled");
    if(
      requestFP16Storage && info.supportsFP16Storage && info.supportsShaderRoundingModeRTEFloat16 &&
      !VulkanShaders::hasRteVariants() && logger != nullptr)
      logger->write(
        string("Vulkan: ") + info.properties.deviceName +
        " supports shaderRoundingModeRTEFloat16, but this build did not compile RTE shader variants");
    return set;
  }

  // The feature structs chained onto the device create info's pNext, rooted at
  // storageFeatures. Each node is populated only when the corresponding feature
  // is actually enabled and the chain is built front-to-back, so the pNext links
  // match the extension ordering of the enabled feature set. The chain must
  // outlive vkCreateDevice, hence the struct owning the nodes.
  struct DeviceFeatureChain {
    VkPhysicalDevice16BitStorageFeatures storageFeatures = {};
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroupFeatures = {};
#endif
#if defined(KATAGO_HAS_SHADER_FLOAT16_INT8_HEADERS)
    VkPhysicalDeviceShaderFloat16Int8Features f16Features = {};
#endif
    VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE_local dot2Features = {};
    CoopmatFeaturesKHR coopmatFeatures = {};
    VulkanMemoryModelFeatures memModelFeatures = {};
    Coopmat2FeaturesNV coopmat2Features = {};
    CoopmatMaintenance1FeaturesEXT maintenance1Features = {};
#if defined(VK_KHR_pipeline_executable_properties)
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExecutableFeatures = {};
#endif

    VkPhysicalDevice16BitStorageFeatures*
    build(const VulkanDeviceInfo& info, bool requestFP16Storage, const DeviceExtensionSet& extSet) {
      // FP16-storage 16-bit access feature.
      storageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
      storageFeatures.storageBuffer16BitAccess = (requestFP16Storage && info.supportsFP16Storage) ? VK_TRUE : VK_FALSE;
      storageFeatures.pNext = nullptr;
#if defined(KATAGO_HAS_SUBGROUP_SIZE_CONTROL_HEADERS)
      subgroupFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
      subgroupFeatures.subgroupSizeControl = info.supportsSubgroupSizeControl ? VK_TRUE : VK_FALSE;
      subgroupFeatures.computeFullSubgroups = info.supportsComputeFullSubgroups ? VK_TRUE : VK_FALSE;
      subgroupFeatures.pNext = nullptr;
      if(subgroupFeatures.subgroupSizeControl == VK_TRUE || subgroupFeatures.computeFullSubgroups == VK_TRUE)
        storageFeatures.pNext = &subgroupFeatures;
#endif

#if defined(KATAGO_HAS_SHADER_FLOAT16_INT8_HEADERS)
      if(extSet.enableFP16Compute) {
        f16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        f16Features.pNext = storageFeatures.pNext;
        f16Features.shaderFloat16 = VK_TRUE;
        storageFeatures.pNext = &f16Features;
      }
#endif

      // VK_VALVE_shader_mixed_float_dot_product feature. The struct type is
      // declared in the file-scope anonymous namespace at the top of this file.
      if(extSet.enableDot2F16 || extSet.enableDot2F16AccF16) {
        dot2Features.sType = kVkStructureTypeShaderMixedFloatDotProductFeaturesVALVE;
        dot2Features.pNext = storageFeatures.pNext;
        dot2Features.shaderMixedFloatDotProductFloat16AccFloat32 = extSet.enableDot2F16 ? VK_TRUE : VK_FALSE;
        dot2Features.shaderMixedFloatDotProductFloat16AccFloat16 = extSet.enableDot2F16AccF16 ? VK_TRUE : VK_FALSE;
        storageFeatures.pNext = &dot2Features;
      }

      // VK_KHR_cooperative_matrix feature. Type declared in the file-scope
      // anonymous namespace at the top of this file (or the real header type).
      if(extSet.enableKhrCoopmatBase) {
        coopmatFeatures.sType = kSTypeCoopmatFeatures;
        coopmatFeatures.pNext = storageFeatures.pNext;
        coopmatFeatures.cooperativeMatrix = VK_TRUE;
        storageFeatures.pNext = &coopmatFeatures;

        // Enable vulkanMemoryModel — the coopmat shaders' OpCapability
        // VulkanMemoryModel is invalid without it. Core-promoted from 1.2 but the
        // feature must still be explicitly enabled; the extension struct type is
        // reused on 1.0/1.1 (with VK_KHR_vulkan_memory_model in the ext list).
        memModelFeatures.sType = kSTypeVulkanMemoryModelFeatures;
        memModelFeatures.pNext = storageFeatures.pNext;
        memModelFeatures.vulkanMemoryModel = VK_TRUE;
        storageFeatures.pNext = &memModelFeatures;
      }

      if(extSet.enableCoopmat2F16 || extSet.enableCoopmat2F16AccF16) {
        coopmat2Features.sType = kSTypeCoopmat2Features;
        coopmat2Features.pNext = storageFeatures.pNext;
        // Tensor addressing is required by every coopmat2 shader and guaranteed
        // present by the discovery gate. The attention-only features are enabled
        // whenever the driver reports them so the coopmat2 attention pipeline is
        // valid when selected; enabling a bit the driver lacks would fail device
        // creation, so unsupported bits stay VK_FALSE.
        coopmat2Features.cooperativeMatrixWorkgroupScope = VK_TRUE;
        coopmat2Features.cooperativeMatrixFlexibleDimensions = VK_TRUE;
        coopmat2Features.cooperativeMatrixTensorAddressing = VK_TRUE;
        coopmat2Features.cooperativeMatrixReductions = info.supportsCoopmat2Reductions ? VK_TRUE : VK_FALSE;
        coopmat2Features.cooperativeMatrixConversions = info.supportsCoopmat2Conversions ? VK_TRUE : VK_FALSE;
        coopmat2Features.cooperativeMatrixPerElementOperations =
          info.supportsCoopmat2PerElementOps ? VK_TRUE : VK_FALSE;
        coopmat2Features.cooperativeMatrixBlockLoads = info.supportsCoopmat2BlockLoads ? VK_TRUE : VK_FALSE;
        storageFeatures.pNext = &coopmat2Features;
      }

      if(extSet.enableCoopmatMaintenance1) {
        maintenance1Features.sType = kSTypeCoopmatMaintenance1Features;
        maintenance1Features.pNext = storageFeatures.pNext;
        maintenance1Features.cooperativeMatrixReductions = VK_TRUE;
        maintenance1Features.cooperativeMatrixConversions = VK_TRUE;
        maintenance1Features.cooperativeMatrixPerElementOperations = VK_TRUE;
        storageFeatures.pNext = &maintenance1Features;
      }

#if defined(VK_KHR_pipeline_executable_properties)
      if(extSet.enablePipelineExecutableStats) {
        pipelineExecutableFeatures.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        pipelineExecutableFeatures.pNext = storageFeatures.pNext;
        pipelineExecutableFeatures.pipelineExecutableInfo = VK_TRUE;
        storageFeatures.pNext = &pipelineExecutableFeatures;
      }
#endif
      return &storageFeatures;
    }
  };

  static InitializedVulkanDevice*
  createDevice(const VulkanDeviceInfo& info, bool requestFP16Storage, bool requestFP16Compute, Logger* logger) {
    auto* dev = new InitializedVulkanDevice(info);

    const DeviceExtensionSet extSet = buildDeviceExtensionSet(info, requestFP16Storage, requestFP16Compute, logger);
    const vector<const char*>& extensions = extSet.names;

    // Reflect what was actually enabled at vkCreateDevice so downstream
    // ctx.supportsDot2F16 matches the driver state (see resolveGemmStridedVariant).
    dev->info.supportsFP16Compute = extSet.enableFP16Compute;
    dev->info.supportsDot2F16 = extSet.enableDot2F16;
    dev->info.supportsDot2F16AccF16 = extSet.enableDot2F16AccF16;
    dev->info.supportsKhrCoopmatBase = extSet.enableKhrCoopmatBase;
    dev->info.supportsCoopmat1F16 = extSet.enableCoopmat1F16;
    dev->info.supportsCoopmat1F16AccF16 = extSet.enableCoopmat1F16AccF16;
    dev->info.supportsCoopmatMaintenance1 = extSet.enableCoopmatMaintenance1;
    dev->info.supportsCoopmat2F16 = extSet.enableCoopmat2F16;
    dev->info.supportsCoopmat2F16AccF16 = extSet.enableCoopmat2F16AccF16;
    dev->info.supportsCoopmat2Attention = info.supportsCoopmat2Attention && extSet.enableFP16Compute;
    if(!extSet.enableCoopmat1F16)
      dev->info.coopmatShapes.clear();
    if(!extSet.enableCoopmat1F16AccF16)
      dev->info.coopmatAccF16Shapes.clear();
    if(!extSet.enableCoopmat2F16)
      dev->info.coopmat2FlexShapes.clear();
    if(!extSet.enableCoopmat2F16AccF16)
      dev->info.coopmat2AccF16FlexShapes.clear();
    if(!extSet.enableCoopmat2F16)
      dev->info.supportsCoopmat2Attention = false;

    const uint32_t computeQueueCount = std::max(1, std::min(4, info.computeQueueCount));
    vector<float> computeQueuePriorities(computeQueueCount, 1.0f);
    vector<float> transferQueuePriorities(1, 1.0f);
    vector<VkDeviceQueueCreateInfo> queueCIs;
    queueCIs.reserve(2);

    VkDeviceQueueCreateInfo computeQueueCI = {};
    computeQueueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueCI.queueFamilyIndex = static_cast<uint32_t>(info.computeQueueFamilyIdx);
    computeQueueCI.queueCount = computeQueueCount;
    computeQueueCI.pQueuePriorities = computeQueuePriorities.data();
    queueCIs.push_back(computeQueueCI);

    if(info.transferQueueFamilyIdx != info.computeQueueFamilyIdx) {
      VkDeviceQueueCreateInfo transferQueueCI = {};
      transferQueueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      transferQueueCI.queueFamilyIndex = static_cast<uint32_t>(info.transferQueueFamilyIdx);
      transferQueueCI.queueCount = 1;
      transferQueueCI.pQueuePriorities = transferQueuePriorities.data();
      queueCIs.push_back(transferQueueCI);
    }

    DeviceFeatureChain featureChain;
    VkPhysicalDevice16BitStorageFeatures* deviceFeatures = featureChain.build(info, requestFP16Storage, extSet);

    VkDeviceCreateInfo deviceCI = {};
    deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.pNext = deviceFeatures;
    deviceCI.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
    deviceCI.pQueueCreateInfos = queueCIs.data();
    deviceCI.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    deviceCI.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateDevice(info.physicalDevice, &deviceCI, nullptr, &dev->device));
    registerShaderRteF16Device(dev->device, extSet.enableShaderRoundingModeRTEFloat16);
    registerPipelineExecutableStatsDevice(dev->device, extSet.enablePipelineExecutableStats);
    dev->computeQueues.resize(computeQueueCount, VK_NULL_HANDLE);
    for(uint32_t q = 0; q < computeQueueCount; q++)
      vkGetDeviceQueue(dev->device, static_cast<uint32_t>(info.computeQueueFamilyIdx), q, &dev->computeQueues[q]);
    dev->computeQueue = dev->computeQueues[0];
    dev->extraComputeQueueMutexes.clear();
    dev->extraComputeQueueMutexes.reserve(computeQueueCount > 0 ? computeQueueCount - 1 : 0);
    for(uint32_t q = 1; q < computeQueueCount; q++)
      dev->extraComputeQueueMutexes.emplace_back(std::make_unique<std::mutex>());

    if(info.transferQueueFamilyIdx == info.computeQueueFamilyIdx) {
      dev->transferQueue = dev->computeQueue;
      dev->transferQueueMutex = &dev->queueMutex;
    } else {
      vkGetDeviceQueue(dev->device, static_cast<uint32_t>(info.transferQueueFamilyIdx), 0, &dev->transferQueue);
      dev->dedicatedTransferQueueMutex = std::make_unique<std::mutex>();
      dev->transferQueueMutex = dev->dedicatedTransferQueueMutex.get();
    }

    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpCI.queueFamilyIndex = static_cast<uint32_t>(info.computeQueueFamilyIdx);
    VK_CHECK(vkCreateCommandPool(dev->device, &cpCI, nullptr, &dev->commandPool));

    VkCommandPoolCreateInfo transferCpCI = {};
    transferCpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transferCpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    transferCpCI.queueFamilyIndex = static_cast<uint32_t>(info.transferQueueFamilyIdx);
    VK_CHECK(vkCreateCommandPool(dev->device, &transferCpCI, nullptr, &dev->transferCommandPool));

    if(logger) {
      logger->write(
        string("Vulkan: initialized device [") + to_string(info.gpuIdx) + "] " + info.properties.deviceName);
      logger->write(
        string("Vulkan: acceleration features on ") + info.properties.deviceName + ": " +
        vulkanAccelerationFeatureSummary(dev->info));
      if(extSet.enablePipelineExecutableStats)
        logger->write("Vulkan: pipeline executable statistics enabled (KATAGO_VULKAN_PIPELINE_STATS=1)");
      if(extSet.enableShaderRoundingModeRTEFloat16)
        logger->write("Vulkan: RoundingModeRTE enabled for FP16 shader conversions");
    }

    return dev;
  }

}  // namespace

VulkanDevicesContext::VulkanDevicesContext(
  const vector<VulkanDeviceInfo>& allDeviceInfos,
  const vector<int>& gpuIdxsToUse,
  bool useFP16Storage,
  bool useFP16Compute,
  Logger* logger)
  : defaultGpuIdx(-1) {
  if(allDeviceInfos.empty())
    throw StringError("No Vulkan compute devices found");

  // Find the best default device
  int bestDesirability = -1;
  for(auto& info: allDeviceInfos) {
    if(info.defaultDesirability > bestDesirability) {
      bestDesirability = info.defaultDesirability;
      defaultGpuIdx = info.gpuIdx;
    }
  }

  // Determine which GPU indices to use
  vector<int> idxsToCreate;
  for(int idx: gpuIdxsToUse) {
    int resolved = (idx == -1) ? defaultGpuIdx : idx;
    bool already = false;
    for(int x: idxsToCreate)
      if(x == resolved) {
        already = true;
        break;
      }
    if(!already)
      idxsToCreate.push_back(resolved);
  }
  if(idxsToCreate.empty())
    idxsToCreate.push_back(defaultGpuIdx);

  for(int idx: idxsToCreate) {
    const VulkanDeviceInfo* found = nullptr;
    for(auto& info: allDeviceInfos)
      if(info.gpuIdx == idx) {
        found = &info;
        break;
      }
    if(!found)
      throw StringError("Vulkan: no device with gpuIdx " + to_string(idx));

    InitializedVulkanDevice* dev = createDevice(*found, useFP16Storage, useFP16Compute, logger);
    devicesToUse.push_back(std::unique_ptr<InitializedVulkanDevice>(dev));

    bool nameExists = false;
    for(auto& n: uniqueDeviceNamesToUse)
      if(n == found->properties.deviceName) {
        nameExists = true;
        break;
      }
    if(!nameExists)
      uniqueDeviceNamesToUse.push_back(found->properties.deviceName);
  }
}

VulkanDevicesContext::~VulkanDevicesContext() = default;

const InitializedVulkanDevice* VulkanDevicesContext::findGpuExn(int gpuIdx) const {
  int resolved = (gpuIdx == -1) ? defaultGpuIdx : gpuIdx;
  for(auto& dev: devicesToUse)
    if(dev->info.gpuIdx == resolved)
      return dev.get();
  throw StringError("Vulkan: device with gpuIdx " + to_string(gpuIdx) + " was not initialized");
}

// printDevices

void VulkanHelpers::printDevices(VkInstance instance, Logger* logger) {
  auto infos = VulkanDeviceInfo::getAllDeviceInfosOnSystem(instance, logger);
  cout << "Vulkan devices (" << infos.size() << "):" << endl;
  for(auto& info: infos) {
    string dtype;
    switch(info.properties.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        dtype = "Discrete GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        dtype = "Integrated GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        dtype = "Virtual GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        dtype = "CPU";
        break;
      default:
        dtype = "Other";
        break;
    }
    cout << "  [" << info.gpuIdx << "] " << info.properties.deviceName << " (" << dtype << ")"
         << (info.supportsFP16Compute ? " FP16" : "") << (info.supportsFP16Storage ? " FP16Storage" : "")
         << (info.supportsShaderRoundingModeRTEFloat16 ? " RTEFloat16" : "")
         << (info.supportsDot2F16 ? " dot2_f16acc32" : "") << (info.supportsDot2F16AccF16 ? " dot2_f16acc16" : "");
    if(info.supportsCoopmat1F16)
      cout << " coopmat(" << info.coopmatShapes.size() << " shapes)";
    if(info.supportsCoopmat1F16AccF16)
      cout << " coopmat_accf16(" << info.coopmatAccF16Shapes.size() << " shapes)";
    if(info.supportsCoopmat2F16)
      cout << " coopmat2(" << info.coopmat2FlexShapes.size() << " flex)";
    if(info.supportsCoopmat2F16AccF16)
      cout << " coopmat2_accf16(" << info.coopmat2AccF16FlexShapes.size() << " flex)";
    if(info.supportsSubgroupShuffleCompute)
      cout << " subgroupShuffle(size=" << info.subgroupSize << ")";
    if(info.canRequireComputeSubgroupSize(info.subgroupSize))
      cout << " subgroupSizeControl(range=" << info.minSubgroupSize << "-" << info.maxSubgroupSize << ")";
    cout << endl;
  }
}

// makeDeviceBuf / makeDeviceBufFP32

VBuf makeDeviceBuf(
  VkDevice device,
  const VkPhysicalDeviceMemoryProperties& memProps,
  size_t numElts,
  bool fp16,
  VkMemoryPropertyFlags memFlags) {
  VkDeviceSize sz = numElts * (fp16 ? sizeof(half_t) : sizeof(float));
  auto buf = VulkanHelpers::allocateBuffer(
    device,
    memProps,
    sz,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    memFlags);
  return std::make_unique<VulkanBuffer>(std::move(buf));
}

VBuf makeDeviceBufFP32(
  VkDevice device,
  const VkPhysicalDeviceMemoryProperties& memProps,
  size_t numElts,
  VkMemoryPropertyFlags memFlags) {
  return makeDeviceBuf(device, memProps, numElts, false, memFlags);
}

// PlannedBufferAllocator

PlannedBufferAllocator::PlannedBufferAllocator(VkDevice dev, const VkPhysicalDeviceMemoryProperties& memProps_)
  : device(dev), memProps(memProps_), built(false) {}

PlannedBufferAllocator::~PlannedBufferAllocator() {
  for(VkDeviceMemory memory: slabMemories) {
    if(memory != VK_NULL_HANDLE)
      vkFreeMemory(device, memory, nullptr);
  }
}

VkDeviceSize PlannedBufferAllocator::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  if(alignment <= 1)
    return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

VkBuffer PlannedBufferAllocator::createBuffer(VkDeviceSize sizeBytes, VkBufferUsageFlags usage) const {
  VkBufferCreateInfo bufInfo = {};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = sizeBytes;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VK_CHECK(vkCreateBuffer(device, &bufInfo, nullptr, &buffer));
  return buffer;
}

void PlannedBufferAllocator::add(
  VBuf* dst,
  VkDeviceSize sizeBytes,
  VkBufferUsageFlags usage,
  VkMemoryPropertyFlags memFlags) {
  Request req = {};
  req.dst = dst;
  req.sizeBytes = sizeBytes;
  req.usage = usage;
  req.memFlags = memFlags;
  requests.push_back(req);
}

void PlannedBufferAllocator::build() {
  if(requests.empty())
    return;
  if(built)
    throw StringError("Vulkan: PlannedBufferAllocator::build called more than once");

  struct BuiltReq {
    VBuf* dst;
    VkBuffer buffer;
    VkDeviceSize sizeBytes;
    VkMemoryRequirements memReqs;
    uint32_t memTypeIndex;
    VkDeviceSize offset;
  };

  struct Bucket {
    VkDeviceSize totalSize;
    VkDeviceMemory memory;
    std::vector<size_t> reqIndices;
    Bucket() : totalSize(0), memory(VK_NULL_HANDLE), reqIndices() {}
  };

  std::vector<BuiltReq> builtReqs;
  std::vector<Bucket> buckets(memProps.memoryTypeCount);
  builtReqs.reserve(requests.size());

  try {
    for(const Request& req: requests) {
      BuiltReq builtReq = {};
      builtReq.dst = req.dst;
      builtReq.sizeBytes = req.sizeBytes;
      builtReq.buffer = createBuffer(req.sizeBytes, req.usage);
      vkGetBufferMemoryRequirements(device, builtReq.buffer, &builtReq.memReqs);
      builtReq.memTypeIndex = findMemoryType(memProps, builtReq.memReqs.memoryTypeBits, req.memFlags);
      builtReq.offset = 0;
      builtReqs.push_back(builtReq);
    }

    for(size_t i = 0; i < builtReqs.size(); i++) {
      BuiltReq& req = builtReqs[i];
      Bucket& bucket = buckets[req.memTypeIndex];
      bucket.totalSize = alignUp(bucket.totalSize, req.memReqs.alignment);
      req.offset = bucket.totalSize;
      bucket.totalSize += req.memReqs.size;
      bucket.reqIndices.push_back(i);
    }

    for(uint32_t memTypeIndex = 0; memTypeIndex < memProps.memoryTypeCount; memTypeIndex++) {
      Bucket& bucket = buckets[memTypeIndex];
      if(bucket.reqIndices.empty())
        continue;

      VkMemoryAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = bucket.totalSize;
      allocInfo.memoryTypeIndex = memTypeIndex;
      VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bucket.memory));
      slabMemories.push_back(bucket.memory);
    }

    for(Bucket& bucket: buckets) {
      if(bucket.reqIndices.empty())
        continue;
      for(size_t reqIdx: bucket.reqIndices) {
        BuiltReq& req = builtReqs[reqIdx];
        VK_CHECK(vkBindBufferMemory(device, req.buffer, bucket.memory, req.offset));
      }
    }

    for(BuiltReq& req: builtReqs) {
      *(req.dst) =
        std::make_unique<VulkanBuffer>(req.buffer, buckets[req.memTypeIndex].memory, req.sizeBytes, device, false);
      req.buffer = VK_NULL_HANDLE;
    }
    built = true;
  } catch(...) {
    for(BuiltReq& req: builtReqs) {
      if(req.buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, req.buffer, nullptr);
    }
    for(VkDeviceMemory memory: slabMemories) {
      if(memory != VK_NULL_HANDLE)
        vkFreeMemory(device, memory, nullptr);
    }
    slabMemories.clear();
    throw;
  }
}

// WeightUploadBatch + WeightUploadBatchGuard

namespace {
  thread_local WeightUploadBatch* g_weightUploadBatch = nullptr;

  // Helper: round up to a multiple.
  size_t roundUpToMultiple(size_t size, size_t ofThis) {
    return ((size + ofThis - 1) / ofThis) * ofThis;
  }
}  // namespace

WeightUploadBatch::WeightUploadBatch(
  VkDevice dev,
  VkQueue q,
  mutex& qMutex,
  VkCommandPool cmdPool,
  const VkPhysicalDeviceMemoryProperties& memProps)
  : device(dev),
    queue(q),
    queueMutex(qMutex),
    commandPool(cmdPool),
    stagingMapped(nullptr),
    stagingCapacity((VkDeviceSize)(64u << 20)),
    stagingUsed(0),
    commandBuffer(VK_NULL_HANDLE),
    fence(VK_NULL_HANDLE),
    hasPendingCopies(false) {
  staging = std::make_unique<VulkanBuffer>(VulkanHelpers::allocateBuffer(
    device,
    memProps,
    stagingCapacity,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  VK_CHECK(vkMapMemory(device, staging->memory, 0, stagingCapacity, 0, &stagingMapped));

  VkCommandBufferAllocateInfo cbAllocInfo = {};
  cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAllocInfo.commandPool = commandPool;
  cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAllocInfo.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAllocInfo, &commandBuffer));

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

  beginRecording();
}

WeightUploadBatch::~WeightUploadBatch() {
  if(commandBuffer != VK_NULL_HANDLE)
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  if(fence != VK_NULL_HANDLE)
    vkDestroyFence(device, fence, nullptr);
  if(stagingMapped != nullptr)
    vkUnmapMemory(device, staging->memory);
}

void WeightUploadBatch::beginRecording() {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
  stagingUsed = 0;
  hasPendingCopies = false;
}

void WeightUploadBatch::submitAndWaitIfNeeded() {
  if(!hasPendingCopies)
    return;
  VK_CHECK(vkEndCommandBuffer(commandBuffer));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  {
    lock_guard<mutex> lock(queueMutex);
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
  }
  VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
  VK_CHECK(vkResetFences(device, 1, &fence));
  VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
  beginRecording();
}

VulkanBuffer WeightUploadBatch::uploadData(
  const void* data,
  VkDeviceSize size,
  const VkPhysicalDeviceMemoryProperties& memProps) {
  if(size > stagingCapacity) {
    submitAndWaitIfNeeded();
    return allocateAndUploadBuffer(device, queue, queueMutex, commandPool, memProps, data, size);
  }
  if(stagingUsed + size > stagingCapacity)
    submitAndWaitIfNeeded();

  // Weights are always consumed as storage buffers by the shaders.
  VulkanBuffer devBuf = VulkanHelpers::allocateBuffer(
    device,
    memProps,
    size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  memcpy(static_cast<char*>(stagingMapped) + stagingUsed, data, static_cast<size_t>(size));
  VkBufferCopy region = {};
  region.srcOffset = stagingUsed;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(commandBuffer, staging->buffer, devBuf.buffer, 1, &region);

  stagingUsed += (VkDeviceSize)roundUpToMultiple(static_cast<size_t>(size), static_cast<size_t>(16));
  hasPendingCopies = true;
  return devBuf;
}

void WeightUploadBatch::finish() {
  submitAndWaitIfNeeded();
}

WeightUploadBatchGuard::WeightUploadBatchGuard(WeightUploadBatch* batch) {
  g_weightUploadBatch = batch;
}

WeightUploadBatchGuard::~WeightUploadBatchGuard() {
  g_weightUploadBatch = nullptr;
}

static size_t winogradCoopmatPackedTileStrideWords(int freeDim, int k, int blockFree, int bk, int padWords) {
  testAssert(freeDim >= 0 && k >= 0 && blockFree > 0 && bk > 0);
  testAssert(padWords >= 0);
  testAssert(freeDim % 4 == 0 && blockFree % 4 == 0);
  const size_t freeTiles = (size_t)((freeDim + blockFree - 1) / blockFree);
  const size_t kBlocks = (size_t)((k + bk - 1) / bk);
  const size_t rowWords = (size_t)(blockFree / 4) + (size_t)padWords;
  return freeTiles * kBlocks * (size_t)bk * rowWords;
}

size_t winogradPackedRowMajorAStrideWords(int m, int k, int bm, int bk, int padWords) {
  testAssert(m >= 0 && k >= 0 && bm > 0 && bk > 0);
  testAssert(padWords >= 0);
  testAssert(bk % 4 == 0);
  const size_t mTiles = (size_t)((m + bm - 1) / bm);
  const size_t kBlocks = (size_t)((k + bk - 1) / bk);
  const size_t rowWords = (size_t)(bk / 4) + (size_t)padWords;
  return mTiles * kBlocks * (size_t)bm * rowWords;
}

size_t winogradCoopmatPackedBStrideWords(int n, int k, int bn, int bk, int padWords) {
  return winogradCoopmatPackedTileStrideWords(n, k, bn, bk, padWords);
}

static std::vector<float> packWinogradCoopmatTileData(
  const std::vector<float>& data,
  int numBatches,
  int freeDim,
  int k,
  int blockFree,
  int bk,
  int padWords) {
  testAssert(numBatches >= 0 && freeDim >= 0 && k >= 0 && blockFree > 0 && bk > 0);
  testAssert(padWords >= 0);
  testAssert(freeDim % 4 == 0 && blockFree % 4 == 0);
  testAssert(data.size() == (size_t)numBatches * (size_t)k * (size_t)freeDim);

  const int freeTiles = (freeDim + blockFree - 1) / blockFree;
  const int kBlocks = (k + bk - 1) / bk;
  const int rowWords = blockFree / 4 + padWords;
  const size_t batchStrideWords = winogradCoopmatPackedTileStrideWords(freeDim, k, blockFree, bk, padWords);

  std::vector<float> packed(batchStrideWords * (size_t)numBatches * 4, 0.0f);
  for(int batch = 0; batch < numBatches; batch++) {
    for(int freeTile = 0; freeTile < freeTiles; freeTile++) {
      for(int kBlock = 0; kBlock < kBlocks; kBlock++) {
        const size_t tileBaseWords =
          ((size_t)batch * (size_t)freeTiles * (size_t)kBlocks + (size_t)freeTile * (size_t)kBlocks + (size_t)kBlock) *
          (size_t)bk * (size_t)rowWords;
        for(int kk = 0; kk < bk; kk++) {
          const int gk = kBlock * bk + kk;
          if(gk >= k)
            continue;
          for(int freeWord = 0; freeWord < blockFree / 4; freeWord++) {
            const int gFreeBase = freeTile * blockFree + freeWord * 4;
            const size_t dstBase = (tileBaseWords + (size_t)kk * (size_t)rowWords + (size_t)freeWord) * 4;
            for(int lane = 0; lane < 4; lane++) {
              const int gFree = gFreeBase + lane;
              if(gFree < freeDim)
                packed[dstBase + (size_t)lane] =
                  data[((size_t)batch * (size_t)k + (size_t)gk) * (size_t)freeDim + (size_t)gFree];
            }
          }
        }
      }
    }
  }
  return packed;
}

std::vector<float>
packWinogradRowMajorAData(const std::vector<float>& data, int numBatches, int m, int k, int bm, int bk, int padWords) {
  testAssert(numBatches >= 0 && m >= 0 && k >= 0 && bm > 0 && bk > 0);
  testAssert(padWords >= 0);
  testAssert(bk % 4 == 0);
  testAssert(data.size() == (size_t)numBatches * (size_t)k * (size_t)m);

  const int mTiles = (m + bm - 1) / bm;
  const int kBlocks = (k + bk - 1) / bk;
  const int rowWords = bk / 4 + padWords;
  const size_t batchStrideWords = winogradPackedRowMajorAStrideWords(m, k, bm, bk, padWords);

  std::vector<float> packed(batchStrideWords * (size_t)numBatches * 4, 0.0f);
  for(int batch = 0; batch < numBatches; batch++) {
    for(int mTile = 0; mTile < mTiles; mTile++) {
      for(int kBlock = 0; kBlock < kBlocks; kBlock++) {
        const size_t tileBaseWords =
          ((size_t)batch * (size_t)mTiles * (size_t)kBlocks + (size_t)mTile * (size_t)kBlocks + (size_t)kBlock) *
          (size_t)bm * (size_t)rowWords;
        for(int mInTile = 0; mInTile < bm; mInTile++) {
          const int gm = mTile * bm + mInTile;
          if(gm >= m)
            continue;
          for(int kkWord = 0; kkWord < bk / 4; kkWord++) {
            const int kkBase = kBlock * bk + kkWord * 4;
            const size_t dstBase = (tileBaseWords + (size_t)mInTile * (size_t)rowWords + (size_t)kkWord) * 4;
            for(int lane = 0; lane < 4; lane++) {
              const int gk = kkBase + lane;
              if(gk < k)
                packed[dstBase + (size_t)lane] =
                  data[((size_t)batch * (size_t)k + (size_t)gk) * (size_t)m + (size_t)gm];
            }
          }
        }
      }
    }
  }
  return packed;
}

std::vector<float> packWinogradCoopmatBWeights(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padWords) {
  return packWinogradCoopmatTileData(weights, numBatches, n, k, bn, bk, padWords);
}

static size_t stridedGemmPackedBStrideVec4s(int n, int k, int bn, int bk, int padScalars) {
  testAssert(n >= 0 && k >= 0 && bn > 0 && bk > 0);
  testAssert(padScalars >= 0);
  testAssert(n % 4 == 0 && bn % 4 == 0);
  testAssert((bk + padScalars) % 4 == 0);
  const size_t nTiles = (size_t)((n + bn - 1) / bn);
  const size_t kBlocks = (size_t)((k + bk - 1) / bk);
  const size_t tileScalars = (size_t)bn * (size_t)(bk + padScalars);
  testAssert(tileScalars % 4 == 0);
  return nTiles * kBlocks * (tileScalars / 4);
}

std::vector<float> packStridedGemmBWeights(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padScalars) {
  testAssert(numBatches >= 0 && n >= 0 && k >= 0 && bn > 0 && bk > 0);
  testAssert(padScalars >= 0);
  testAssert(n % 4 == 0 && bn % 4 == 0);
  testAssert((bk + padScalars) % 4 == 0);
  testAssert(weights.size() == (size_t)numBatches * (size_t)k * (size_t)n);

  const int nTiles = (n + bn - 1) / bn;
  const int kBlocks = (k + bk - 1) / bk;
  const int rowStride = bk + padScalars;
  const size_t batchStrideVec4s = stridedGemmPackedBStrideVec4s(n, k, bn, bk, padScalars);

  std::vector<float> packed(batchStrideVec4s * (size_t)numBatches * 4, 0.0f);
  for(int batch = 0; batch < numBatches; batch++) {
    for(int nTile = 0; nTile < nTiles; nTile++) {
      for(int kBlock = 0; kBlock < kBlocks; kBlock++) {
        const size_t tileBase =
          ((size_t)batch * (size_t)nTiles * (size_t)kBlocks + (size_t)nTile * (size_t)kBlocks + (size_t)kBlock) *
          (size_t)bn * (size_t)rowStride;
        for(int nLocal = 0; nLocal < bn; nLocal++) {
          const int gn = nTile * bn + nLocal;
          if(gn >= n)
            continue;
          for(int kLocal = 0; kLocal < bk; kLocal++) {
            const int gk = kBlock * bk + kLocal;
            if(gk < k)
              packed[tileBase + (size_t)nLocal * (size_t)rowStride + (size_t)kLocal] =
                weights[((size_t)batch * (size_t)k + (size_t)gk) * (size_t)n + (size_t)gn];
          }
        }
      }
    }
  }
  return packed;
}

std::vector<float> packStridedGemmBWeightsRowMajor(
  const std::vector<float>& weights,
  int numBatches,
  int n,
  int k,
  int bn,
  int bk,
  int padScalars) {
  testAssert(numBatches >= 0 && n >= 0 && k >= 0 && bn > 0 && bk > 0);
  testAssert(padScalars >= 0);
  testAssert(n % 4 == 0 && bn % 4 == 0);
  testAssert((bn + padScalars) % 4 == 0);
  testAssert(weights.size() == (size_t)numBatches * (size_t)k * (size_t)n);

  const int nTiles = (n + bn - 1) / bn;
  const int kBlocks = (k + bk - 1) / bk;
  const int rowStride = bn + padScalars;
  const size_t tileScalars = (size_t)bk * (size_t)rowStride;
  std::vector<float> packed((size_t)numBatches * (size_t)nTiles * (size_t)kBlocks * tileScalars, 0.0f);
  for(int batch = 0; batch < numBatches; batch++) {
    for(int nTile = 0; nTile < nTiles; nTile++) {
      for(int kBlock = 0; kBlock < kBlocks; kBlock++) {
        const size_t tileBase =
          ((size_t)batch * (size_t)nTiles * (size_t)kBlocks + (size_t)nTile * (size_t)kBlocks + (size_t)kBlock) *
          tileScalars;
        for(int kLocal = 0; kLocal < bk; kLocal++) {
          const int gk = kBlock * bk + kLocal;
          if(gk >= k)
            continue;
          for(int nLocal = 0; nLocal < bn; nLocal++) {
            const int gn = nTile * bn + nLocal;
            if(gn < n)
              packed[tileBase + (size_t)kLocal * (size_t)rowStride + (size_t)nLocal] =
                weights[((size_t)batch * (size_t)k + (size_t)gk) * (size_t)n + (size_t)gn];
          }
        }
      }
    }
  }
  return packed;
}

// makeWeightBuf

VBuf makeWeightBuf(
  VkDevice device,
  VkQueue queue,
  mutex& queueMutex,
  VkCommandPool commandPool,
  const VkPhysicalDeviceMemoryProperties& memProps,
  const vector<float>& weights,
  bool fp16) {
  if(weights.empty())
    return nullptr;
  if(!fp16) {
    VulkanBuffer buf =
      (g_weightUploadBatch != nullptr)
        ? g_weightUploadBatch->uploadData(weights.data(), weights.size() * sizeof(float), memProps)
        : allocateAndUploadBuffer(
            device, queue, queueMutex, commandPool, memProps, weights.data(), weights.size() * sizeof(float));
    return std::make_unique<VulkanBuffer>(std::move(buf));
  } else {
    vector<half_t> half_weights(weights.size());
    for(size_t i = 0; i < weights.size(); i++)
      half_weights[i] = half_float::half_cast<half_t>(weights[i]);
    VulkanBuffer buf =
      (g_weightUploadBatch != nullptr)
        ? g_weightUploadBatch->uploadData(
            half_weights.data(), half_weights.size() * sizeof(half_t), memProps)
        : allocateAndUploadBuffer(
            device,
            queue,
            queueMutex,
            commandPool,
            memProps,
            half_weights.data(),
            half_weights.size() * sizeof(half_t));
    return std::make_unique<VulkanBuffer>(std::move(buf));
  }
}

// ScratchBuffers

ScratchBuffers::ScratchBuffers(
  VkDevice dev,
  const VkPhysicalDeviceMemoryProperties& mp,
  bool fp16_,
  int pxyLen,
  int maxBatch)
  : device(dev), memProps(mp), fp16(fp16_), paddedNNXYLen(pxyLen), maxBatchSize(maxBatch) {
  size_t elemSize = fp16_ ? 2u : 4u;
  batchXYBytes = static_cast<size_t>(maxBatch) * pxyLen * elemSize;
  batchXYFloatBytes = static_cast<size_t>(maxBatch) * pxyLen * 4;
  batchFloatBytes = static_cast<size_t>(maxBatch) * 4;
  batchBytes = static_cast<size_t>(maxBatch) * elemSize;
  VkDevice dev_ = dev;
  VkPhysicalDeviceMemoryProperties mp_ = mp;
  allocator = std::make_unique<SimpleAllocator<VulkanBuffer*>>(
    [dev_, mp_](size_t sz) -> VulkanBuffer* {
      auto b = VulkanHelpers::allocateBuffer(
        dev_,
        mp_,
        sz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      return new VulkanBuffer(std::move(b));
    },
    [](VulkanBuffer* p) noexcept { delete p; });
}

void ScratchBuffers::setPermanentSlots(const vector<VkDeviceSize>& slotSizes) {
  permanentSlots.resize(slotSizes.size());
  for(size_t i = 0; i < slotSizes.size(); i++) {
    VkDeviceSize sz = std::max(slotSizes[i], (VkDeviceSize)1);
    auto buf = VulkanHelpers::allocateBuffer(
      device,
      memProps,
      sz,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    permanentSlots[i] = std::make_unique<VulkanBuffer>(std::move(buf));
  }
}

VulkanBuffer* ScratchBuffers::permanentSlot(int i) const {
  testAssert(i >= 0 && static_cast<size_t>(i) < permanentSlots.size());
  return permanentSlots[i].get();
}

#endif  // USE_VULKAN_BACKEND

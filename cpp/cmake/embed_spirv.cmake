# embed_spirv.cmake
# Called by CMake -P to embed listed .spv files into
# OUTPUT_CPP and OUTPUT_H as C++ uint32_t arrays.
#
# Variables expected (passed via -D):
#   SPV_MANIFEST - newline-separated list of .spv files to embed
#   OUTPUT_CPP   - path for generated .cpp
#   OUTPUT_H     - path for generated .h

if(NOT SPV_MANIFEST)
  message(FATAL_ERROR "embed_spirv.cmake: SPV_MANIFEST not set")
endif()
if(NOT OUTPUT_CPP)
  message(FATAL_ERROR "embed_spirv.cmake: OUTPUT_CPP not set")
endif()
if(NOT OUTPUT_H)
  message(FATAL_ERROR "embed_spirv.cmake: OUTPUT_H not set")
endif()

file(STRINGS "${SPV_MANIFEST}" SPV_FILES)

set(H_CONTENT "#pragma once\n#include <cstddef>\n#include <cstdint>\n\nnamespace VulkanShaders {\n")
set(CPP_CONTENT "#include \"vulkanshaders_generated.h\"\n\nnamespace VulkanShaders {\n")

set(ALL_VAR_NAMES "")
set(RTE_NORMAL_VAR_NAMES "")

foreach(SPV_FILE ${SPV_FILES})
  get_filename_component(BASE_NAME "${SPV_FILE}" NAME_WE)
  # Replace dashes and dots with underscores for a valid C identifier
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" VAR_NAME "${BASE_NAME}")
  list(APPEND ALL_VAR_NAMES "${VAR_NAME}")
  if(VAR_NAME MATCHES "_rte$")
    string(REGEX REPLACE "_rte$" "" NORMAL_VAR_NAME "${VAR_NAME}")
    list(APPEND RTE_NORMAL_VAR_NAMES "${NORMAL_VAR_NAME}")
  endif()

  file(READ "${SPV_FILE}" HEX_CONTENT HEX)
  string(LENGTH "${HEX_CONTENT}" HEX_LEN)

  # Convert hex string to comma-separated uint32_t hex literals (4 bytes each)
  math(EXPR NUM_BYTES "${HEX_LEN} / 2")
  math(EXPR NUM_WORDS "(${NUM_BYTES} + 3) / 4")

  set(WORD_LIST "")
  math(EXPR LAST_BYTE_IDX "${NUM_BYTES} - 1")
  set(WORD_IDX 0)
  while(WORD_IDX LESS NUM_WORDS)
    math(EXPR BYTE0_IDX "${WORD_IDX} * 4")
    math(EXPR B0 "${BYTE0_IDX} * 2")
    math(EXPR B1 "${B0} + 2")
    math(EXPR B2 "${B0} + 4")
    math(EXPR B3 "${B0} + 6")

    # Extract each byte of the word (little-endian in SPIR-V)
    if(B0 LESS ${HEX_LEN})
      string(SUBSTRING "${HEX_CONTENT}" ${B0} 2 BYTE0)
    else()
      set(BYTE0 "00")
    endif()
    if(B1 LESS ${HEX_LEN})
      string(SUBSTRING "${HEX_CONTENT}" ${B1} 2 BYTE1)
    else()
      set(BYTE1 "00")
    endif()
    if(B2 LESS ${HEX_LEN})
      string(SUBSTRING "${HEX_CONTENT}" ${B2} 2 BYTE2)
    else()
      set(BYTE2 "00")
    endif()
    if(B3 LESS ${HEX_LEN})
      string(SUBSTRING "${HEX_CONTENT}" ${B3} 2 BYTE3)
    else()
      set(BYTE3 "00")
    endif()

    list(APPEND WORD_LIST "0x${BYTE3}${BYTE2}${BYTE1}${BYTE0}u")
    math(EXPR WORD_IDX "${WORD_IDX} + 1")
  endwhile()

  list(JOIN WORD_LIST ", " WORD_LIST_STR)

  string(APPEND H_CONTENT
    "  extern const uint32_t ${VAR_NAME}[${NUM_WORDS}];\n"
    "  extern const uint32_t ${VAR_NAME}_size; // number of uint32_t words\n"
  )

  string(APPEND CPP_CONTENT
    "  const uint32_t ${VAR_NAME}[${NUM_WORDS}] = { ${WORD_LIST_STR} };\n"
    "  const uint32_t ${VAR_NAME}_size = ${NUM_WORDS};\n"
  )
endforeach()

string(APPEND H_CONTENT
  "\n"
  "  bool hasRteVariants();\n"
  "  bool getRteVariant(const uint32_t* spirvData, size_t spirvWords, const uint32_t*& rteData, size_t& rteWords);\n"
)

set(HAS_RTE_VARIANTS "false")
set(RTE_LOOKUP_BODY "")
foreach(NORMAL_VAR_NAME ${RTE_NORMAL_VAR_NAMES})
  set(RTE_VAR_NAME "${NORMAL_VAR_NAME}_rte")
  list(FIND ALL_VAR_NAMES "${NORMAL_VAR_NAME}" NORMAL_IDX)
  list(FIND ALL_VAR_NAMES "${RTE_VAR_NAME}" RTE_IDX)
  if(NOT NORMAL_IDX EQUAL -1 AND NOT RTE_IDX EQUAL -1)
    set(HAS_RTE_VARIANTS "true")
    string(APPEND RTE_LOOKUP_BODY
      "    if(spirvData == ${NORMAL_VAR_NAME} && spirvWords == ${NORMAL_VAR_NAME}_size) {\n"
      "      rteData = ${RTE_VAR_NAME};\n"
      "      rteWords = ${RTE_VAR_NAME}_size;\n"
      "      return true;\n"
      "    }\n"
    )
  endif()
endforeach()

string(APPEND CPP_CONTENT
  "\n"
  "  bool hasRteVariants() {\n"
  "    return ${HAS_RTE_VARIANTS};\n"
  "  }\n"
  "\n"
  "  bool getRteVariant(const uint32_t* spirvData, size_t spirvWords, const uint32_t*& rteData, size_t& rteWords) {\n"
  "    rteData = nullptr;\n"
  "    rteWords = 0;\n"
)
if(RTE_LOOKUP_BODY STREQUAL "")
  string(APPEND CPP_CONTENT
    "    (void)spirvData;\n"
    "    (void)spirvWords;\n"
  )
else()
  string(APPEND CPP_CONTENT "${RTE_LOOKUP_BODY}")
endif()
string(APPEND CPP_CONTENT
  "    return false;\n"
  "  }\n"
)

string(APPEND H_CONTENT "} // namespace VulkanShaders\n")
string(APPEND CPP_CONTENT "} // namespace VulkanShaders\n")

file(WRITE "${OUTPUT_H}" "${H_CONTENT}")
file(WRITE "${OUTPUT_CPP}" "${CPP_CONTENT}")

message(STATUS "embed_spirv.cmake: generated ${OUTPUT_CPP} and ${OUTPUT_H}")

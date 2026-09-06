# The two skymizer scorers: {llm,vlm}-kld. They share the header-only
# skymizer-common.h / skymizer-vlmk-kernel.h; the VLM tool additionally
# links mtmd.
#   llama-vlm-kld    dual-model on-the-fly KLD/JSD metrics (no logit storage)
#   llama-llm-kld    dual-model text-only on-the-fly KLD/JSD metrics
foreach(tool IN ITEMS vlm-kld llm-kld)
    set(target llama-${tool})
    add_executable(${target} ${tool}.cpp)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${target})
    if (LLAMA_TOOLS_INSTALL)
        install(TARGETS ${target} RUNTIME)
    endif()
    if(tool MATCHES "^vlm-")
        target_link_libraries(${target} PRIVATE vendor::hash llama-common mtmd Threads::Threads)
    else()
        target_link_libraries(${target} PRIVATE vendor::hash llama-common Threads::Threads)
    endif()
    target_compile_features(${target} PRIVATE cxx_std_17)
endforeach()

# Batch reference generation uses GGUF templates/tokenization and native sampling.
add_executable(llama-reference reference.cpp)
target_link_libraries(llama-reference PRIVATE vendor::hash llama-common mtmd Threads::Threads)
target_compile_features(llama-reference PRIVATE cxx_std_17)
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-reference RUNTIME)
endif()

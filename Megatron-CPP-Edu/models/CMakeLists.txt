# Models CMakeLists.txt
add_subdirectory(gpt)
add_subdirectory(transformer)

# Create models library
add_library(megatron_models STATIC
    gpt/gpt_model.cpp
    transformer/transformer_classifier.cpp
)

target_link_libraries(megatron_models
    megatron_core
)

target_include_directories(megatron_models
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
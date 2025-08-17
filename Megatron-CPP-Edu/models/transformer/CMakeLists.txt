# Transformer Models CMakeLists.txt
add_library(megatron_transformer
)

target_include_directories(megatron_transformer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_transformer
    PUBLIC megatron_core
    PUBLIC megatron_training
)
# Training CMakeLists.txt
add_library(megatron_training
)

target_include_directories(megatron_training
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_training
    PUBLIC megatron_core
)
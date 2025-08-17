# Distributed CMakeLists.txt
add_library(megatron_distributed
)

target_include_directories(megatron_distributed
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_distributed
    PUBLIC megatron_core
)
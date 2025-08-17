# GPT Models CMakeLists.txt
add_library(megatron_gpt
    gpt_model.cpp
)

target_include_directories(megatron_gpt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_gpt
    PUBLIC megatron_core
)
# GPT Models CMakeLists.txt
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpt_model.cpp")
    add_library(megatron_gpt
        gpt_model.cpp
    )
else()
    # Create empty library for now
    add_library(megatron_gpt gpt_model.cpp)
    target_sources(megatron_gpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/empty.cpp)
endif()

target_include_directories(megatron_gpt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_gpt
    PUBLIC megatron_core
    PUBLIC megatron_training
)
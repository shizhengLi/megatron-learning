# Distributed CMakeLists.txt
add_library(megatron_distributed
    communication.cpp
)

target_include_directories(megatron_distributed
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(megatron_distributed
    PUBLIC megatron_core
    PUBLIC megatron_parallel
)

if(MPI_FOUND)
    target_link_libraries(megatron_distributed
        PUBLIC MPI::MPI_CXX
    )
    target_compile_definitions(megatron_distributed
        PRIVATE USE_MPI
    )
endif()
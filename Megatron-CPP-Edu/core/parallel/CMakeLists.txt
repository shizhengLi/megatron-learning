# Parallel computing module CMakeLists.txt

# Create parallel computing library
add_library(megatron_parallel
    tensor_parallel.cpp
    data_parallel.cpp
)

# Include directories
target_include_directories(megatron_parallel
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link dependencies
target_link_libraries(megatron_parallel
    PUBLIC megatron_core
)

if(MPI_FOUND)
    target_link_libraries(megatron_parallel
        PUBLIC MPI::MPI_CXX
    )
    target_compile_definitions(megatron_parallel
        PRIVATE USE_MPI
    )
endif()

# Installation
install(TARGETS megatron_parallel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
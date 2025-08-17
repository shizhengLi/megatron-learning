add_library(megatron_core
    tensor/tensor.cpp
    layers/layer.cpp
    layers/linear.cpp
    layers/layer_norm.cpp
    layers/dropout.cpp
    layers/embedding.cpp
)

# Include directories
target_include_directories(megatron_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link dependencies
if(Eigen3_FOUND)
    target_link_libraries(megatron_core
        PUBLIC Eigen3::Eigen
    )
    target_compile_definitions(megatron_core
        PRIVATE EIGEN_MPL2_ONLY
    )
endif()

if(MPI_FOUND)
    target_link_libraries(megatron_core
        PUBLIC MPI::MPI_CXX
    )
endif()

if(USE_CUDA)
    target_compile_definitions(megatron_core PRIVATE USE_CUDA)
    target_link_libraries(megatron_core PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Installation
install(TARGETS megatron_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
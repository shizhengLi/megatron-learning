add_library(megatron_core
    tensor/tensor.cpp
    layers/layer.cpp
    layers/linear.cpp
    layers/layer_norm.cpp
    layers/dropout.cpp
    layers/embedding.cpp
    layers/attention.cpp
    layers/transformer_block.cpp
    optimizers/optimizer.cpp
    optimizers/sgd.cpp
    optimizers/adamw.cpp
    loss/cross_entropy_loss.cpp
    training/trainer.cpp
    data/dataset.cpp
    evaluation/metrics.cpp
    performance/performance.cpp
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
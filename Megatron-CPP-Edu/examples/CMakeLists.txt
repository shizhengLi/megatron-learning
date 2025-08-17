# Examples CMakeLists.txt
add_executable(real_world_applications real_world_applications.cpp)
target_link_libraries(real_world_applications
    PRIVATE megatron_core
    PRIVATE megatron_models
)
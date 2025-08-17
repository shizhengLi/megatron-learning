# Examples CMakeLists.txt
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example
    PRIVATE megatron_core
    PRIVATE megatron_training
)
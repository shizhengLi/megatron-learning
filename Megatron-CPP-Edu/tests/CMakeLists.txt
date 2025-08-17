# Tests CMakeLists.txt
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, using custom implementation")
    add_definitions(-DUSE_CUSTOM_GTEST)
endif()

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor 
    PRIVATE megatron_core
)
if(GTest_FOUND)
    target_link_libraries(test_tensor 
        PRIVATE GTest::gtest 
        PRIVATE GTest::gtest_main
    )
endif()

# Add test to CTest
if(GTest_FOUND)
    add_test(NAME test_tensor COMMAND test_tensor)
    # Test discovery
    include(GoogleTest)
    gtest_discover_tests(test_tensor)
else()
    # Custom main function when GTest is not available
    add_executable(test_tensor_main test_tensor_main.cpp)
    target_link_libraries(test_tensor_main PRIVATE megatron_core)
    add_test(NAME test_tensor COMMAND test_tensor_main)
endif()
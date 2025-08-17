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

# Layer tests
add_executable(test_layers test_layers.cpp)
target_link_libraries(test_layers 
    PRIVATE megatron_core
)
if(GTest_FOUND)
    target_link_libraries(test_layers 
        PRIVATE GTest::gtest 
        PRIVATE GTest::gtest_main
    )
endif()

# Add test to CTest
if(GTest_FOUND)
    add_test(NAME test_layers COMMAND test_layers)
    # Test discovery
    include(GoogleTest)
    gtest_discover_tests(test_layers)
else()
    # Custom main function when GTest is not available
    add_executable(test_layers_main test_layers_main.cpp)
    target_link_libraries(test_layers_main PRIVATE megatron_core)
    add_test(NAME test_layers COMMAND test_layers_main)
endif()

# Phase 3 tests
add_executable(test_phase3 test_phase3.cpp)
target_link_libraries(test_phase3 
    PRIVATE megatron_core
)
if(GTest_FOUND)
    target_link_libraries(test_phase3 
        PRIVATE GTest::gtest 
        PRIVATE GTest::gtest_main
    )
endif()

# Add test to CTest
if(GTest_FOUND)
    add_test(NAME test_phase3 COMMAND test_phase3)
    # Test discovery
    include(GoogleTest)
    gtest_discover_tests(test_phase3)
else()
    # Custom main function when GTest is not available
    add_executable(test_phase3_main test_phase3_main.cpp)
    target_link_libraries(test_phase3_main PRIVATE megatron_core)
    add_test(NAME test_phase3 COMMAND test_phase3_main)
endif()

# Phase 4 tests
add_executable(test_phase4 test_phase4.cpp)
target_link_libraries(test_phase4 
    PRIVATE megatron_core
    PRIVATE megatron_models
)
if(GTest_FOUND)
    target_link_libraries(test_phase4 
        PRIVATE GTest::gtest 
        PRIVATE GTest::gtest_main
    )
endif()

# Add test to CTest
if(GTest_FOUND)
    add_test(NAME test_phase4 COMMAND test_phase4)
    # Test discovery
    include(GoogleTest)
    gtest_discover_tests(test_phase4)
else()
    # Custom main function when GTest is not available
    add_executable(test_phase4_main test_phase4_main.cpp)
    target_link_libraries(test_phase4_main PRIVATE megatron_core megatron_models)
    add_test(NAME test_phase4 COMMAND test_phase4_main)
endif()
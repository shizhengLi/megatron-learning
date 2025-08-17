#include <iostream>
#include <cassert>
#include "../core/tensor/tensor.h"

using namespace megatron;

void test_basic_construction() {
    std::cout << "Testing basic construction..." << std::endl;
    
    Tensor t1({2, 3});
    assert(t1.shape().size() == 2);
    assert(t1.shape()[0] == 2);
    assert(t1.shape()[1] == 3);
    assert(t1.size() == 6);
    assert(t1.dim() == 2);
    
    Tensor t2({4, 5, 6});
    assert(t2.shape().size() == 3);
    assert(t2.shape()[0] == 4);
    assert(t2.shape()[1] == 5);
    assert(t2.shape()[2] == 6);
    assert(t2.size() == 120);
    
    std::cout << "✓ Basic construction tests passed" << std::endl;
}

void test_fill_operations() {
    std::cout << "Testing fill operations..." << std::endl;
    
    Tensor t({3, 4});
    
    t.zeros();
    for (int i = 0; i < t.size(); ++i) {
        assert(t[i] == 0.0f);
    }
    
    t.ones();
    for (int i = 0; i < t.size(); ++i) {
        assert(t[i] == 1.0f);
    }
    
    t.fill(5.0f);
    for (int i = 0; i < t.size(); ++i) {
        assert(t[i] == 5.0f);
    }
    
    std::cout << "✓ Fill operations tests passed" << std::endl;
}

void test_arithmetic_operations() {
    std::cout << "Testing arithmetic operations..." << std::endl;
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(3.0f);
    b.fill(2.0f);
    
    Tensor c = a + b;
    for (int i = 0; i < c.size(); ++i) {
        assert(c[i] == 5.0f);
    }
    
    Tensor d = a - b;
    for (int i = 0; i < d.size(); ++i) {
        assert(d[i] == 1.0f);
    }
    
    Tensor e = a * b;
    for (int i = 0; i < e.size(); ++i) {
        assert(e[i] == 6.0f);
    }
    
    Tensor f = a / b;
    for (int i = 0; i < f.size(); ++i) {
        assert(f[i] == 1.5f);
    }
    
    std::cout << "✓ Arithmetic operations tests passed" << std::endl;
}

void test_matrix_multiplication() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    Tensor a({2, 3});
    Tensor b({3, 2});
    
    a.fill(1.0f);
    b.fill(1.0f);
    
    Tensor c = a.matmul(b);
    
    assert(c.shape().size() == 2);
    assert(c.shape()[0] == 2);
    assert(c.shape()[1] == 2);
    
    // Each element should be 3 (1*1 + 1*1 + 1*1)
    for (int i = 0; i < c.size(); ++i) {
        assert(c[i] == 3.0f);
    }
    
    std::cout << "✓ Matrix multiplication tests passed" << std::endl;
}

void test_activation_functions() {
    std::cout << "Testing activation functions..." << std::endl;
    
    Tensor x({3});
    x[0] = -1.0f;
    x[1] = 0.0f;
    x[2] = 1.0f;
    
    Tensor relu = x.relu();
    assert(relu[0] == 0.0f);
    assert(relu[1] == 0.0f);
    assert(relu[2] == 1.0f);
    
    Tensor sigmoid = x.sigmoid();
    assert(sigmoid[0] > 0.0f && sigmoid[0] < 0.5f);
    assert(sigmoid[1] == 0.5f);
    assert(sigmoid[2] > 0.5f && sigmoid[2] < 1.0f);
    
    std::cout << "✓ Activation functions tests passed" << std::endl;
}

void test_transpose() {
    std::cout << "Testing transpose..." << std::endl;
    
    Tensor t({2, 3});
    for (int i = 0; i < t.size(); ++i) {
        t[i] = i + 1;
    }
    
    Tensor transposed = t.transpose();
    
    assert(transposed.shape().size() == 2);
    assert(transposed.shape()[0] == 3);
    assert(transposed.shape()[1] == 2);
    
    // Check that transposed values are correct
    assert(transposed[0] == t[0]);  // t[0,0] -> transposed[0,0]
    assert(transposed[1] == t[3]);  // t[1,0] -> transposed[0,1]
    assert(transposed[2] == t[1]);  // t[0,1] -> transposed[1,0]
    assert(transposed[3] == t[4]);  // t[1,1] -> transposed[1,1]
    
    std::cout << "✓ Transpose tests passed" << std::endl;
}

void test_reshape() {
    std::cout << "Testing reshape..." << std::endl;
    
    Tensor t({2, 3, 4});
    for (int i = 0; i < t.size(); ++i) {
        t[i] = i + 1;
    }
    
    t.reshape({4, 6});
    assert(t.shape().size() == 2);
    assert(t.shape()[0] == 4);
    assert(t.shape()[1] == 6);
    assert(t.size() == 24);
    
    std::cout << "✓ Reshape tests passed" << std::endl;
}

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;
    
    bool exception_thrown = false;
    
    try {
        Tensor t({0, 2});
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    exception_thrown = false;
    try {
        Tensor a({2, 3});
        Tensor b({3, 2});
        Tensor c = a + b;
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "✓ Error handling tests passed" << std::endl;
}

int main() {
    std::cout << "Running Tensor Tests..." << std::endl;
    
    test_basic_construction();
    test_fill_operations();
    test_arithmetic_operations();
    test_matrix_multiplication();
    test_activation_functions();
    test_transpose();
    test_reshape();
    test_error_handling();
    
    std::cout << "\n🎉 All tensor tests passed!" << std::endl;
    return 0;
}
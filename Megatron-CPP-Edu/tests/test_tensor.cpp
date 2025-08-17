#include <gtest/gtest.h>
#include "core/tensor/tensor.h"
#include <iostream>
#include <cmath>

using namespace megatron;

TEST(TensorTest, BasicConstruction) {
    // Test basic construction
    Tensor t1({2, 3});
    EXPECT_EQ(t1.shape().size(), 2);
    EXPECT_EQ(t1.shape()[0], 2);
    EXPECT_EQ(t1.shape()[1], 3);
    EXPECT_EQ(t1.size(), 6);
    EXPECT_EQ(t1.dim(), 2);
    
    // Test initializer list construction
    Tensor t2({4, 5, 6});
    EXPECT_EQ(t2.shape().size(), 3);
    EXPECT_EQ(t2.shape()[0], 4);
    EXPECT_EQ(t2.shape()[1], 5);
    EXPECT_EQ(t2.shape()[2], 6);
    EXPECT_EQ(t2.size(), 120);
}

TEST(TensorTest, BasicOperations) {
    Tensor a({2, 3});
    Tensor b({2, 3});
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    Tensor c = a + b;
    
    EXPECT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    
    for (int i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
}

TEST(TensorTest, FillOperations) {
    Tensor t({3, 4});
    
    t.zeros();
    for (int i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
    
    t.ones();
    for (int i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 1.0f);
    }
    
    t.fill(5.0f);
    for (int i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 5.0f);
    }
}

TEST(TensorTest, RandomNormal) {
    Tensor t({100, 100});
    t.random_normal(0.0f, 1.0f);
    
    // Check that values are approximately normally distributed
    float mean = 0.0f;
    float variance = 0.0f;
    
    for (int i = 0; i < t.size(); ++i) {
        mean += t[i];
    }
    mean /= t.size();
    
    for (int i = 0; i < t.size(); ++i) {
        variance += (t[i] - mean) * (t[i] - mean);
    }
    variance /= t.size();
    
    // Mean should be close to 0, variance close to 1
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(variance, 1.0f, 0.2f);
}

TEST(TensorTest, ArithmeticOperations) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(3.0f);
    b.fill(2.0f);
    
    Tensor c = a + b;
    for (int i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c[i], 5.0f);
    }
    
    Tensor d = a - b;
    for (int i = 0; i < d.size(); ++i) {
        EXPECT_FLOAT_EQ(d[i], 1.0f);
    }
    
    Tensor e = a * b;
    for (int i = 0; i < e.size(); ++i) {
        EXPECT_FLOAT_EQ(e[i], 6.0f);
    }
    
    Tensor f = a / b;
    for (int i = 0; i < f.size(); ++i) {
        EXPECT_FLOAT_EQ(f[i], 1.5f);
    }
}

TEST(TensorTest, MatrixMultiplication) {
    Tensor a({2, 3});
    Tensor b({3, 4});
    
    // Fill a with row-wise values
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            a[i * 3 + j] = i * 3 + j + 1; // 1,2,3,4,5,6
        }
    }
    
    // Fill b with column-wise values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            b[i * 4 + j] = i * 4 + j + 1; // 1,2,3,4,5,6,7,8,9,10,11,12
        }
    }
    
    Tensor c = a.matmul(b);
    
    EXPECT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 4);
    
    // Check some values
    // First row: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    EXPECT_FLOAT_EQ(c[0], 38.0f); // 1+10+27
    EXPECT_FLOAT_EQ(c[1], 44.0f); // 2+12+30
    EXPECT_FLOAT_EQ(c[2], 50.0f); // 3+14+33
    EXPECT_FLOAT_EQ(c[3], 56.0f); // 4+16+36
    
    // Second row: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
    EXPECT_FLOAT_EQ(c[4], 83.0f); // 4+25+54
    EXPECT_FLOAT_EQ(c[5], 98.0f); // 8+30+60
    EXPECT_FLOAT_EQ(c[6], 113.0f); // 12+35+66
    EXPECT_FLOAT_EQ(c[7], 128.0f); // 16+40+72
}

TEST(TensorTest, SumAndMean) {
    Tensor t({2, 3});
    t.fill(2.0f);
    
    // Test sum all elements
    Tensor sum_all = t.sum();
    EXPECT_EQ(sum_all.shape().size(), 1);
    EXPECT_EQ(sum_all.shape()[0], 1);
    EXPECT_FLOAT_EQ(sum_all[0], 12.0f); // 6 elements * 2.0
    
    // Test sum along rows (dim=0)
    Tensor sum_rows = t.sum(0);
    EXPECT_EQ(sum_rows.shape().size(), 1);
    EXPECT_EQ(sum_rows.shape()[0], 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(sum_rows[i], 4.0f); // 2 elements * 2.0
    }
    
    // Test sum along columns (dim=1)
    Tensor sum_cols = t.sum(1);
    EXPECT_EQ(sum_cols.shape().size(), 1);
    EXPECT_EQ(sum_cols.shape()[0], 2);
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(sum_cols[i], 6.0f); // 3 elements * 2.0
    }
    
    // Test mean
    Tensor mean_all = t.mean();
    EXPECT_FLOAT_EQ(mean_all[0], 2.0f);
    
    Tensor mean_rows = t.mean(0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(mean_rows[i], 2.0f);
    }
}

TEST(TensorTest, MaxOperation) {
    Tensor t({2, 3});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f;
    t[3] = 4.0f; t[4] = 5.0f; t[5] = 6.0f;
    
    // Test max all elements
    Tensor max_all = t.max();
    EXPECT_FLOAT_EQ(max_all[0], 6.0f);
    
    // Test max along rows (dim=0)
    Tensor max_rows = t.max(0);
    EXPECT_EQ(max_rows.shape().size(), 1);
    EXPECT_EQ(max_rows.shape()[0], 3);
    EXPECT_FLOAT_EQ(max_rows[0], 4.0f); // max(1,4)
    EXPECT_FLOAT_EQ(max_rows[1], 5.0f); // max(2,5)
    EXPECT_FLOAT_EQ(max_rows[2], 6.0f); // max(3,6)
    
    // Test max along columns (dim=1)
    Tensor max_cols = t.max(1);
    EXPECT_EQ(max_cols.shape().size(), 1);
    EXPECT_EQ(max_cols.shape()[0], 2);
    EXPECT_FLOAT_EQ(max_cols[0], 3.0f); // max(1,2,3)
    EXPECT_FLOAT_EQ(max_cols[1], 6.0f); // max(4,5,6)
}

TEST(TensorTest, ActivationFunctions) {
    Tensor x({5});
    x[0] = -2.0f; x[1] = -1.0f; x[2] = 0.0f; x[3] = 1.0f; x[4] = 2.0f;
    
    // Test ReLU
    Tensor relu = x.relu();
    EXPECT_FLOAT_EQ(relu[0], 0.0f);
    EXPECT_FLOAT_EQ(relu[1], 0.0f);
    EXPECT_FLOAT_EQ(relu[2], 0.0f);
    EXPECT_FLOAT_EQ(relu[3], 1.0f);
    EXPECT_FLOAT_EQ(relu[4], 2.0f);
    
    // Test Sigmoid
    Tensor sigmoid = x.sigmoid();
    EXPECT_NEAR(sigmoid[0], 0.1192f, 1e-4f);
    EXPECT_NEAR(sigmoid[1], 0.2689f, 1e-4f);
    EXPECT_NEAR(sigmoid[2], 0.5f, 1e-4f);
    EXPECT_NEAR(sigmoid[3], 0.7311f, 1e-4f);
    EXPECT_NEAR(sigmoid[4], 0.8808f, 1e-4f);
    
    // Test Tanh
    Tensor tanh = x.tanh();
    EXPECT_NEAR(tanh[0], -0.9640f, 1e-4f);
    EXPECT_NEAR(tanh[1], -0.7616f, 1e-4f);
    EXPECT_NEAR(tanh[2], 0.0f, 1e-4f);
    EXPECT_NEAR(tanh[3], 0.7616f, 1e-4f);
    EXPECT_NEAR(tanh[4], 0.9640f, 1e-4f);
    
    // Test GELU
    Tensor gelu = x.gelu();
    EXPECT_NEAR(gelu[0], -0.0454f, 1e-3f);
    EXPECT_NEAR(gelu[1], -0.1588f, 1e-3f);
    EXPECT_NEAR(gelu[2], 0.0f, 1e-3f);
    EXPECT_NEAR(gelu[3], 0.8412f, 1e-3f);
    EXPECT_NEAR(gelu[4], 1.9546f, 1e-3f);
}

TEST(TensorTest, Softmax) {
    Tensor t({2, 3});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f;
    t[3] = 4.0f; t[4] = 5.0f; t[5] = 6.0f;
    
    Tensor softmax = t.softmax(1);
    
    // Check that softmax values sum to 1 for each row
    float row1_sum = softmax[0] + softmax[1] + softmax[2];
    float row2_sum = softmax[3] + softmax[4] + softmax[5];
    
    EXPECT_NEAR(row1_sum, 1.0f, 1e-6f);
    EXPECT_NEAR(row2_sum, 1.0f, 1e-6f);
    
    // Check that values are positive and decreasing
    EXPECT_GT(softmax[0], 0.0f);
    EXPECT_GT(softmax[1], 0.0f);
    EXPECT_GT(softmax[2], 0.0f);
    EXPECT_GT(softmax[3], 0.0f);
    EXPECT_GT(softmax[4], 0.0f);
    EXPECT_GT(softmax[5], 0.0f);
    
    EXPECT_LT(softmax[0], softmax[1]);
    EXPECT_LT(softmax[1], softmax[2]);
    EXPECT_LT(softmax[3], softmax[4]);
    EXPECT_LT(softmax[4], softmax[5]);
}

TEST(TensorTest, Transpose) {
    Tensor t({2, 3});
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            t[i * 3 + j] = i * 3 + j + 1; // 1,2,3,4,5,6
        }
    }
    
    Tensor transposed = t.transpose();
    
    EXPECT_EQ(transposed.shape().size(), 2);
    EXPECT_EQ(transposed.shape()[0], 3);
    EXPECT_EQ(transposed.shape()[1], 2);
    
    // Check values
    EXPECT_FLOAT_EQ(transposed[0], 1.0f); // t[0,0]
    EXPECT_FLOAT_EQ(transposed[1], 4.0f); // t[1,0]
    EXPECT_FLOAT_EQ(transposed[2], 2.0f); // t[0,1]
    EXPECT_FLOAT_EQ(transposed[3], 5.0f); // t[1,1]
    EXPECT_FLOAT_EQ(transposed[4], 3.0f); // t[0,2]
    EXPECT_FLOAT_EQ(transposed[5], 6.0f); // t[1,2]
}

TEST(TensorTest, Slice) {
    Tensor t({3, 4});
    for (int i = 0; i < 12; ++i) {
        t[i] = i + 1; // 1 to 12
    }
    
    // Test slicing rows
    Tensor rows = t.slice(0, 1, 3);
    EXPECT_EQ(rows.shape().size(), 2);
    EXPECT_EQ(rows.shape()[0], 2); // 3-1=2 rows
    EXPECT_EQ(rows.shape()[1], 4);
    
    // Check values
    EXPECT_FLOAT_EQ(rows[0], 5.0f); // t[1,0]
    EXPECT_FLOAT_EQ(rows[1], 6.0f); // t[1,1]
    EXPECT_FLOAT_EQ(rows[4], 9.0f); // t[2,0]
    EXPECT_FLOAT_EQ(rows[5], 10.0f); // t[2,1]
    
    // Test slicing columns
    Tensor cols = t.slice(1, 1, 3);
    EXPECT_EQ(cols.shape().size(), 2);
    EXPECT_EQ(cols.shape()[0], 3);
    EXPECT_EQ(cols.shape()[1], 2); // 3-1=2 columns
    
    // Check values
    EXPECT_FLOAT_EQ(cols[0], 2.0f); // t[0,1]
    EXPECT_FLOAT_EQ(cols[1], 3.0f); // t[0,2]
    EXPECT_FLOAT_EQ(cols[2], 6.0f); // t[1,1]
    EXPECT_FLOAT_EQ(cols[3], 7.0f); // t[1,2]
}

TEST(TensorTest, ReshapeAndView) {
    Tensor t({2, 3, 4});
    for (int i = 0; i < t.size(); ++i) {
        t[i] = i + 1;
    }
    
    // Test reshape
    t.reshape({4, 6});
    EXPECT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 6);
    EXPECT_EQ(t.size(), 24);
    
    // Test view
    Tensor view = t.view({6, 4});
    EXPECT_EQ(view.shape().size(), 2);
    EXPECT_EQ(view.shape()[0], 6);
    EXPECT_EQ(view.shape()[1], 4);
    EXPECT_EQ(view.size(), 24);
    
    // Values should be the same
    for (int i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(view[i], t[i]);
    }
}

TEST(TensorTest, CopyAndMove) {
    Tensor original({2, 2});
    original.fill(5.0f);
    
    // Test copy constructor
    Tensor copy(original);
    EXPECT_EQ(copy.shape(), original.shape());
    for (int i = 0; i < copy.size(); ++i) {
        EXPECT_FLOAT_EQ(copy[i], original[i]);
    }
    
    // Test copy assignment
    Tensor assigned({1, 1});
    assigned = original;
    EXPECT_EQ(assigned.shape(), original.shape());
    for (int i = 0; i < assigned.size(); ++i) {
        EXPECT_FLOAT_EQ(assigned[i], original[i]);
    }
    
    // Test move constructor
    Tensor moved(std::move(original));
    EXPECT_EQ(moved.shape().size(), 2);
    EXPECT_EQ(moved.shape()[0], 2);
    EXPECT_EQ(moved.shape()[1], 2);
    for (int i = 0; i < moved.size(); ++i) {
        EXPECT_FLOAT_EQ(moved[i], 5.0f);
    }
}

TEST(TensorTest, Concatenate) {
    Tensor a({2, 2});
    Tensor b({2, 3});
    Tensor c({2, 1});
    
    a.fill(1.0f);
    b.fill(2.0f);
    c.fill(3.0f);
    
    Tensor result = concatenate({a, b, c}, 1);
    
    EXPECT_EQ(result.shape().size(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 6); // 2+3+1
    
    // Check values
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(result[i * 6 + j], 1.0f);
        }
        for (int j = 2; j < 5; ++j) {
            EXPECT_FLOAT_EQ(result[i * 6 + j], 2.0f);
        }
        for (int j = 5; j < 6; ++j) {
            EXPECT_FLOAT_EQ(result[i * 6 + j], 3.0f);
        }
    }
}

TEST(TensorTest, ErrorHandling) {
    // Test invalid shape
    EXPECT_THROW(Tensor({0, 2}), std::invalid_argument);
    EXPECT_THROW(Tensor({2, -1}), std::invalid_argument);
    
    // Test invalid operations
    Tensor a({2, 3});
    Tensor b({3, 2});
    
    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a - b, std::invalid_argument);
    EXPECT_THROW(a * b, std::invalid_argument);
    EXPECT_THROW(a / b, std::invalid_argument);
    
    // Test invalid matrix multiplication
    Tensor c({2, 3});
    Tensor d({4, 5});
    EXPECT_THROW(c.matmul(d), std::invalid_argument);
    
    // Test invalid reshape
    EXPECT_THROW(a.reshape({2, 4}), std::invalid_argument); // Wrong size
    
    // Test invalid slice
    EXPECT_THROW(a.slice(0, 1, 4), std::invalid_argument); // End out of bounds
    EXPECT_THROW(a.slice(2, 0, 1), std::invalid_argument); // Invalid dimension
    
    // Test invalid index access
    EXPECT_THROW(a[-1], std::out_of_range);
    EXPECT_THROW(a[6], std::out_of_range);
}

TEST(TensorTest, UtilityFunctions) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    
    a.fill(3.0f);
    b.fill(4.0f);
    
    // Test utility functions
    Tensor sum = add(a, b);
    for (int i = 0; i < sum.size(); ++i) {
        EXPECT_FLOAT_EQ(sum[i], 7.0f);
    }
    
    Tensor product = multiply(a, b);
    for (int i = 0; i < product.size(); ++i) {
        EXPECT_FLOAT_EQ(product[i], 12.0f);
    }
    
    Tensor matrix_product = matmul(a, b);
    for (int i = 0; i < matrix_product.size(); ++i) {
        EXPECT_FLOAT_EQ(matrix_product[i], 24.0f); // 3*4 + 3*4 = 24 for each element
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
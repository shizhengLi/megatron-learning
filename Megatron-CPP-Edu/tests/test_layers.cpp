#include <gtest/gtest.h>
#include "../core/layers/linear.h"
#include "../core/layers/layer_norm.h"
#include "../core/layers/dropout.h"
#include "../core/layers/embedding.h"
#include <iostream>

using namespace megatron;

TEST(LayerTest, LinearLayer) {
    // Test linear layer creation
    Linear layer(3, 2, true, "test_linear");
    
    EXPECT_EQ(layer.name(), "test_linear");
    EXPECT_TRUE(layer.is_training());
    
    // Test forward pass
    Tensor input({2, 3});  // batch_size=2, in_features=3
    input.fill(1.0f);
    
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);  // batch_size
    EXPECT_EQ(output.shape()[1], 2);  // out_features
    
    // Test backward pass
    Tensor grad_output({2, 2});
    grad_output.fill(0.1f);
    
    Tensor grad_input = layer.backward(grad_output);
    EXPECT_EQ(grad_input.shape().size(), 2);
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 3);
    
    // Test parameters and gradients
    auto params = layer.parameters();
    auto grads = layer.gradients();
    
    EXPECT_EQ(params.size(), 2);  // weight and bias
    EXPECT_EQ(grads.size(), 2);   // weight_grad and bias_grad
    
    // Test parameter shapes
    EXPECT_EQ(params[0].shape().size(), 2);  // weight
    EXPECT_EQ(params[0].shape()[0], 2);       // out_features
    EXPECT_EQ(params[0].shape()[1], 3);       // in_features
    
    EXPECT_EQ(params[1].shape().size(), 1);  // bias
    EXPECT_EQ(params[1].shape()[0], 2);      // out_features
    
    // Test mode switching
    layer.eval();
    EXPECT_FALSE(layer.is_training());
    
    layer.train();
    EXPECT_TRUE(layer.is_training());
    
    // Test zero_grad
    layer.zero_grad();
    auto zero_grads = layer.gradients();
    for (int i = 0; i < zero_grads[0].size(); ++i) {
        EXPECT_FLOAT_EQ(zero_grads[0][i], 0.0f);
    }
}

TEST(LayerTest, LayerNorm) {
    // Test layer norm creation
    LayerNorm layer_norm(4, 1e-5f, "test_layer_norm");
    
    EXPECT_EQ(layer_norm.name(), "test_layer_norm");
    
    // Test forward pass
    Tensor input({2, 4});  // batch_size=2, normalized_shape=4
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f; input[3] = 4.0f;
    input[4] = 5.0f; input[5] = 6.0f; input[6] = 7.0f; input[7] = 8.0f;
    
    Tensor output = layer_norm.forward(input);
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
    
    // Test backward pass
    Tensor grad_output({2, 4});
    grad_output.fill(0.1f);
    
    Tensor grad_input = layer_norm.backward(grad_output);
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // Test parameters and gradients
    auto params = layer_norm.parameters();
    auto grads = layer_norm.gradients();
    
    EXPECT_EQ(params.size(), 2);  // weight and bias
    EXPECT_EQ(grads.size(), 2);   // weight_grad and bias_grad
    
    // Test parameter shapes
    EXPECT_EQ(params[0].shape().size(), 1);  // weight
    EXPECT_EQ(params[0].shape()[0], 4);      // normalized_shape
    
    EXPECT_EQ(params[1].shape().size(), 1);  // bias
    EXPECT_EQ(params[1].shape()[0], 4);      // normalized_shape
}

TEST(LayerTest, Dropout) {
    // Test dropout creation
    Dropout dropout(0.5f, "test_dropout");
    
    EXPECT_EQ(dropout.name(), "test_dropout");
    EXPECT_FLOAT_EQ(dropout.get_p(), 0.5f);
    
    // Test forward pass in training mode
    Tensor input({2, 3});
    input.fill(2.0f);
    
    Tensor output = dropout.forward(input);
    EXPECT_EQ(output.shape(), input.shape());
    
    // Some values should be zeroed out (with high probability)
    int zero_count = 0;
    for (int i = 0; i < output.size(); ++i) {
        if (output[i] == 0.0f) {
            zero_count++;
        }
    }
    // In training mode, some values should be dropped
    EXPECT_GT(zero_count, 0);
    
    // Test forward pass in evaluation mode
    dropout.eval();
    Tensor eval_output = dropout.forward(input);
    
    // In eval mode, output should equal input
    for (int i = 0; i < eval_output.size(); ++i) {
        EXPECT_FLOAT_EQ(eval_output[i], input[i]);
    }
    
    // Test backward pass
    dropout.train();
    Tensor grad_output({2, 3});
    grad_output.fill(0.1f);
    
    Tensor grad_input = dropout.backward(grad_output);
    EXPECT_EQ(grad_input.shape(), grad_output.shape());
    
    // Test parameters and gradients (should be empty)
    auto params = dropout.parameters();
    auto grads = dropout.gradients();
    
    EXPECT_TRUE(params.empty());
    EXPECT_TRUE(grads.empty());
    
    // Test setting probability
    dropout.set_p(0.3f);
    EXPECT_FLOAT_EQ(dropout.get_p(), 0.3f);
}

TEST(LayerTest, Embedding) {
    // Test embedding creation
    Embedding embedding(1000, 64, "test_embedding");
    
    EXPECT_EQ(embedding.name(), "test_embedding");
    
    // Test forward pass
    Tensor input({2, 3});  // batch_size=2, seq_len=3
    input[0] = 10; input[1] = 20; input[2] = 30;
    input[3] = 40; input[4] = 50; input[5] = 60;
    
    Tensor output = embedding.forward(input);
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);   // batch_size
    EXPECT_EQ(output.shape()[1], 3);   // seq_len
    EXPECT_EQ(output.shape()[2], 64);  // embedding_dim
    
    // Test backward pass
    Tensor grad_output({2, 3, 64});
    grad_output.fill(0.1f);
    
    Tensor grad_input = embedding.backward(grad_output);
    
    // Test parameters and gradients
    auto params = embedding.parameters();
    auto grads = embedding.gradients();
    
    EXPECT_EQ(params.size(), 1);  // weight
    EXPECT_EQ(grads.size(), 1);  // weight_grad
    
    // Test parameter shapes
    EXPECT_EQ(params[0].shape().size(), 2);  // weight
    EXPECT_EQ(params[0].shape()[0], 1000);   // vocab_size
    EXPECT_EQ(params[0].shape()[1], 64);     // embedding_dim
    
    // Test invalid token index
    Tensor invalid_input({1, 1});
    invalid_input[0] = 2000;  // Out of vocabulary range
    
    EXPECT_THROW(embedding.forward(invalid_input), std::out_of_range);
}

TEST(LayerTest, SequentialLayerOperations) {
    // Test combining multiple layers
    Linear linear1(3, 4, true, "linear1");
    LayerNorm layer_norm(4, 1e-5f, "layer_norm");
    Dropout dropout(0.5f, "dropout");
    Linear linear2(4, 2, true, "linear2");
    
    // Test data flow through layers
    Tensor input({2, 3});
    input.fill(1.0f);
    
    // Forward pass through all layers
    Tensor x = linear1.forward(input);
    x = layer_norm.forward(x);
    x = dropout.forward(x);
    x = linear2.forward(x);
    
    EXPECT_EQ(x.shape().size(), 2);
    EXPECT_EQ(x.shape()[0], 2);
    EXPECT_EQ(x.shape()[1], 2);
    
    // Test backward pass
    Tensor grad_output({2, 2});
    grad_output.fill(0.1f);
    
    Tensor grad = linear2.backward(grad_output);
    grad = dropout.backward(grad);
    grad = layer_norm.backward(grad);
    grad = linear1.backward(grad);
    
    EXPECT_EQ(grad.shape(), input.shape());
}

TEST(LayerTest, LayerTrainingModes) {
    Linear linear(3, 2, true, "test_linear");
    Dropout dropout(0.5f, "test_dropout");
    
    // Test initial state
    EXPECT_TRUE(linear.is_training());
    EXPECT_TRUE(dropout.is_training());
    
    // Test switching to eval mode
    linear.eval();
    dropout.eval();
    
    EXPECT_FALSE(linear.is_training());
    EXPECT_FALSE(dropout.is_training());
    
    // Test switching back to train mode
    linear.train();
    dropout.train();
    
    EXPECT_TRUE(linear.is_training());
    EXPECT_TRUE(dropout.is_training());
}

TEST(LayerTest, ParameterAccess) {
    Linear linear(3, 2, true, "test_linear");
    LayerNorm layer_norm(4, 1e-5f, "test_layer_norm");
    
    // Test linear layer parameter access
    const Tensor& linear_weight = linear.weight();
    const Tensor& linear_bias = linear.bias();
    
    EXPECT_EQ(linear_weight.shape().size(), 2);
    EXPECT_EQ(linear_weight.shape()[0], 2);
    EXPECT_EQ(linear_weight.shape()[1], 3);
    
    EXPECT_EQ(linear_bias.shape().size(), 1);
    EXPECT_EQ(linear_bias.shape()[0], 2);
    
    // Test layer norm parameter access
    const Tensor& norm_weight = layer_norm.weight();
    const Tensor& norm_bias = layer_norm.bias();
    
    EXPECT_EQ(norm_weight.shape().size(), 1);
    EXPECT_EQ(norm_weight.shape()[0], 4);
    
    EXPECT_EQ(norm_bias.shape().size(), 1);
    EXPECT_EQ(norm_bias.shape()[0], 4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
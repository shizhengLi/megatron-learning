#include <iostream>
#include <memory>
#include "models/gpt/gpt_model.h"
#include "core/tensor/tensor.h"

using namespace megatron;

int main() {
    try {
        std::cout << "Testing GPT Model..." << std::endl;
        
        // Create a minimal GPT model
        int vocab_size = 100;
        int max_seq_len = 8;
        int embed_dim = 32;
        int num_heads = 2;
        int num_layers = 1;
        int ff_dim = 64;
        
        auto model = std::make_shared<GPTModel>(
            vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim,
            false, 0.0f, "test_gpt");
        
        // Create input tensor with valid token indices
        Tensor input({1, 4});  // batch_size=1, seq_len=4
        for (int i = 0; i < input.size(); ++i) {
            input[i] = i % vocab_size;  // Ensure tokens are within vocab range
        }
        
        std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
        
        // Forward pass
        Tensor output = model->forward(input);
        
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "✓ GPT Model test passed" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ GPT Model test failed: " << e.what() << std::endl;
        return 1;
    }
}
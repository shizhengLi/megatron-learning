#include <iostream>
#include "core/tensor/tensor.h"
#include "core/layers/embedding.h"

using namespace megatron;

int main() {
    std::cout << "Testing Embedding layer..." << std::endl;
    
    try {
        // Create embedding layer
        Embedding embed(1000, 128, "test_embed");
        
        // Create input tensor
        Tensor input({2, 16});
        std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
        std::cout << "Input size: " << input.size() << std::endl;
        
        // Fill with valid token indices
        for (int i = 0; i < input.size(); ++i) {
            input[i] = i % 1000;  // Ensure tokens are within vocab range
            std::cout << "input[" << i << "] = " << input[i] << std::endl;
        }
        
        // Forward pass
        Tensor output = embed.forward(input);
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "Output size: " << output.size() << std::endl;
        
        std::cout << "✓ Embedding test passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ Embedding test failed: " << e.what() << std::endl;
        return 1;
    }
}